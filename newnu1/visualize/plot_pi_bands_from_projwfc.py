import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


STATE_RE = re.compile(
    r"state #\s*(\d+): atom\s*(\d+)\s*\(([^)]+)\),\s*wfc\s*\d+\s*\(l=\s*(\d+)\s*m=\s*(\d+)\)"
)
K_RE = re.compile(r"^\s*k =\s*([\-0-9Ee.+]+)\s+([\-0-9Ee.+]+)\s+([\-0-9Ee.+]+)")
E_RE = re.compile(r"==== e\(\s*(\d+)\)\s*=\s*([\-0-9Ee.+]+)\s*eV")
PSI_TERM_RE = re.compile(r"([0-9]*\.?[0-9]+)\*\[#\s*(\d+)\]")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parse projwfc output (proj.out) and plot only pi bands "
            "(chosen from p_z projector weight)."
        )
    )
    parser.add_argument("--proj-out", default="proj.out", help="Path to projwfc output file.")
    parser.add_argument(
        "--pi-l",
        type=int,
        default=1,
        help="Angular momentum l for pi-like projectors (default: 1 for p).",
    )
    parser.add_argument(
        "--pi-m",
        type=int,
        default=1,
        help="Magnetic index m treated as p_z in QE output (default: 1).",
    )
    parser.add_argument(
        "--min-pi-weight",
        type=float,
        default=0.45,
        help="Minimum pi weight at a k-point to classify a state as pi-like (default: 0.45).",
    )
    parser.add_argument(
        "--max-pi-bands",
        type=int,
        default=None,
        help="Optional cap on number of selected pi states per k-point.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["per-k", "band-mean"],
        default="per-k",
        help=(
            "Selection strategy: per-k (robust, projection-only at each k-point) "
            "or band-mean (legacy band-index mode). Default: per-k."
        ),
    )
    parser.add_argument(
        "--fermi",
        type=float,
        default=None,
        help="Fermi level in eV. If provided, y-axis is E-Ef.",
    )
    parser.add_argument(
        "--ky",
        type=float,
        default=None,
        help="Target ky for kx cut. If omitted, inferred from minimum pi-band gap.",
    )
    parser.add_argument(
        "--kx",
        type=float,
        default=None,
        help="Target kx for ky cut. If omitted, inferred from minimum pi-band gap.",
    )
    parser.add_argument(
        "--savefig",
        type=str,
        default=None,
        help="Output figure path (default: <proj_stem>_pi_bands.png).",
    )
    return parser.parse_args()


def _finalize_band_record(records, current_k, band_idx, energy_ev, pi_weight):
    if current_k is None or band_idx is None or energy_ev is None:
        return
    records.append(
        {
            "kx": current_k[0],
            "ky": current_k[1],
            "kz": current_k[2],
            "band": band_idx,
            "energy_ev": energy_ev,
            "pi_weight": pi_weight,
        }
    )


def parse_projwfc_output(path, pi_l=1, pi_m=1):
    state_map = {}
    pi_state_ids = set()
    records = []

    current_k = None
    band_idx = None
    energy_ev = None
    pi_weight = 0.0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            state_match = STATE_RE.search(line)
            if state_match:
                sid = int(state_match.group(1))
                l_val = int(state_match.group(4))
                m_val = int(state_match.group(5))
                state_map[sid] = (l_val, m_val)
                if l_val == pi_l and m_val == pi_m:
                    pi_state_ids.add(sid)
                continue

            k_match = K_RE.search(line)
            if k_match:
                _finalize_band_record(records, current_k, band_idx, energy_ev, pi_weight)
                band_idx = None
                energy_ev = None
                pi_weight = 0.0
                current_k = (
                    float(k_match.group(1)),
                    float(k_match.group(2)),
                    float(k_match.group(3)),
                )
                continue

            e_match = E_RE.search(line)
            if e_match:
                _finalize_band_record(records, current_k, band_idx, energy_ev, pi_weight)
                band_idx = int(e_match.group(1))
                energy_ev = float(e_match.group(2))
                pi_weight = 0.0
                continue

            if band_idx is not None and "*[#" in line:
                for coeff_text, sid_text in PSI_TERM_RE.findall(line):
                    sid = int(sid_text)
                    if sid in pi_state_ids:
                        pi_weight += float(coeff_text)

    _finalize_band_record(records, current_k, band_idx, energy_ev, pi_weight)

    if not records:
        raise ValueError("No k-point / band / psi records parsed from projwfc output.")
    if not pi_state_ids:
        raise ValueError(
            f"No projector states found with l={pi_l}, m={pi_m}. "
            "Check --pi-m (QE mapping can vary)."
        )

    return records, pi_state_ids, state_map


def select_pi_bands(records, min_pi_weight, max_pi_bands=None):
    by_band = {}
    for rec in records:
        by_band.setdefault(rec["band"], []).append(rec["pi_weight"])

    band_scores = {
        band: float(np.mean(weights))
        for band, weights in by_band.items()
    }
    selected = sorted([band for band, score in band_scores.items() if score >= min_pi_weight])

    if not selected:
        # Fallback: take the top two bands by mean pi weight.
        selected = [
            band
            for band, _ in sorted(band_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        ]
        selected = sorted(selected)

    if max_pi_bands is not None and max_pi_bands > 0:
        selected = selected[:max_pi_bands]

    return selected, band_scores


def _group_by_k(records):
    by_k = {}
    for rec in records:
        key = (rec["kx"], rec["ky"], rec["kz"])
        by_k.setdefault(key, []).append(rec)
    return by_k


def select_pi_states_per_k(records, min_pi_weight, max_pi_bands=None):
    """
    Projection-only selection at every k-point.
    Returns records annotated with:
      rec["pi_selected"] : bool
      rec["pi_rank"] : 1-based energy rank among selected pi states at that k
    """
    by_k = _group_by_k(records)
    fallback_count = max_pi_bands if (max_pi_bands is not None and max_pi_bands > 0) else 2
    fallback_count = max(1, fallback_count)

    annotated = []
    for _, entries in by_k.items():
        # Start with threshold-based selection.
        selected = [rec for rec in entries if rec["pi_weight"] >= min_pi_weight]
        # Fallback for difficult points: take top-N by pi weight.
        if not selected:
            selected = sorted(entries, key=lambda rec: rec["pi_weight"], reverse=True)[:fallback_count]
        # Optional cap: keep strongest pi-weight states only.
        if max_pi_bands is not None and max_pi_bands > 0 and len(selected) > max_pi_bands:
            selected = sorted(selected, key=lambda rec: rec["pi_weight"], reverse=True)[:max_pi_bands]

        selected_ids = {id(rec) for rec in selected}
        selected_energy_order = sorted(selected, key=lambda rec: rec["energy_ev"])
        rank_by_id = {id(rec): rank + 1 for rank, rec in enumerate(selected_energy_order)}

        for rec in entries:
            enriched = dict(rec)
            if id(rec) in selected_ids:
                enriched["pi_selected"] = True
                enriched["pi_rank"] = rank_by_id[id(rec)]
            else:
                enriched["pi_selected"] = False
                enriched["pi_rank"] = None
            annotated.append(enriched)

    return annotated


def infer_dirac_center_from_per_k(records):
    by_k = _group_by_k(records)
    best = None
    for (kx, ky, _kz), entries in by_k.items():
        sel = sorted([rec for rec in entries if rec.get("pi_selected")], key=lambda rec: rec["energy_ev"])
        if len(sel) < 2:
            continue
        for i in range(len(sel) - 1):
            gap = abs(sel[i + 1]["energy_ev"] - sel[i]["energy_ev"])
            if best is None or gap < best["gap"]:
                best = {
                    "gap": gap,
                    "kx": kx,
                    "ky": ky,
                    "pair": (sel[i]["band"], sel[i + 1]["band"]),
                }
    if best is None:
        raise ValueError("Could not infer a crossing/min-gap point from selected pi states.")
    return best


def infer_dirac_center_band_mean(records, selected_bands):
    by_k = {}
    for rec in records:
        if rec["band"] in selected_bands:
            key = (rec["kx"], rec["ky"], rec["kz"])
            by_k.setdefault(key, {})[rec["band"]] = rec["energy_ev"]

    best = None
    for (kx, ky, _kz), band_energy_map in by_k.items():
        present = sorted((band, en) for band, en in band_energy_map.items())
        if len(present) < 2:
            continue
        energies = [item[1] for item in present]
        bands = [item[0] for item in present]
        for i in range(len(energies) - 1):
            gap = abs(energies[i + 1] - energies[i])
            if best is None or gap < best["gap"]:
                best = {
                    "gap": gap,
                    "kx": kx,
                    "ky": ky,
                    "pair": (bands[i], bands[i + 1]),
                }

    if best is None:
        raise ValueError("Could not infer a crossing/min-gap point from selected pi bands.")
    return best


def nearest_cut(records, fixed_value, along="x"):
    if along == "x":
        fixed_axis = "ky"
        sweep_axis = "kx"
    else:
        fixed_axis = "kx"
        sweep_axis = "ky"

    fixed_vals = np.array(sorted(set(rec[fixed_axis] for rec in records)), dtype=float)
    chosen_fixed = float(fixed_vals[np.argmin(np.abs(fixed_vals - fixed_value))])
    cut_records = [rec for rec in records if abs(rec[fixed_axis] - chosen_fixed) < 1e-10]
    cut_records.sort(key=lambda rec: rec[sweep_axis])
    return chosen_fixed, cut_records


def main():
    args = parse_args()
    if not os.path.exists(args.proj_out):
        print(f"Error: file not found: {args.proj_out}")
        return

    try:
        records, pi_state_ids, _state_map = parse_projwfc_output(
            args.proj_out,
            pi_l=args.pi_l,
            pi_m=args.pi_m,
        )
        if args.selection_mode == "per-k":
            records_for_plot = select_pi_states_per_k(
                records,
                min_pi_weight=args.min_pi_weight,
                max_pi_bands=args.max_pi_bands,
            )
            min_gap_info = infer_dirac_center_from_per_k(records_for_plot)
        else:
            selected_bands, band_scores = select_pi_bands(
                records,
                min_pi_weight=args.min_pi_weight,
                max_pi_bands=args.max_pi_bands,
            )
            records_for_plot = [dict(rec) for rec in records]
            for rec in records_for_plot:
                rec["pi_selected"] = rec["band"] in selected_bands
                rec["pi_rank"] = rec["band"] if rec["pi_selected"] else None
            min_gap_info = infer_dirac_center_band_mean(records, selected_bands)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    kx_target = min_gap_info["kx"] if args.kx is None else args.kx
    ky_target = min_gap_info["ky"] if args.ky is None else args.ky

    ky_cut, kx_cut_records = nearest_cut(records_for_plot, ky_target, along="x")
    kx_cut, ky_cut_records = nearest_cut(records_for_plot, kx_target, along="y")

    def band_curve(cut_records, sweep_axis, rank):
        pts = [
            (rec[sweep_axis], rec["energy_ev"], rec["pi_weight"])
            for rec in cut_records
            if rec.get("pi_selected") and rec.get("pi_rank") == rank
        ]
        if not pts:
            return np.array([]), np.array([]), np.array([])
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        w = np.array([p[2] for p in pts], dtype=float)
        return x, y, w

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange", "tab:brown"]
    fermi = args.fermi

    ylabel = r"$E - E_F$ (eV)" if fermi is not None else r"$E$ (eV)"

    if args.selection_mode == "per-k":
        rank_ids = sorted(
            {
                rec["pi_rank"]
                for rec in records_for_plot
                if rec.get("pi_selected") and rec.get("pi_rank") is not None
            }
        )
    else:
        rank_ids = sorted(
            {
                rec["pi_rank"]
                for rec in records_for_plot
                if rec.get("pi_selected") and rec.get("pi_rank") is not None
            }
        )

    rank_stats = {}
    for rank in rank_ids:
        weights = [
            rec["pi_weight"]
            for rec in records_for_plot
            if rec.get("pi_selected") and rec.get("pi_rank") == rank
        ]
        rank_stats[rank] = float(np.mean(weights)) if weights else 0.0

    for j, rank in enumerate(rank_ids):
        color = colors[j % len(colors)]
        x1, y1, w1 = band_curve(kx_cut_records, "kx", rank)
        x2, y2, w2 = band_curve(ky_cut_records, "ky", rank)
        if x1.size == 0 and x2.size == 0:
            continue

        if fermi is not None:
            y1 = y1 - fermi
            y2 = y2 - fermi

        if args.selection_mode == "per-k":
            label = f"pi rank {rank} (mean w_pi={rank_stats[rank]:.3f})"
        else:
            label = f"Band {rank} (mean w_pi={rank_stats[rank]:.3f})"
        if x1.size > 0:
            axes[0].scatter(x1, y1, s=18, color=color, label=label)
        if x2.size > 0:
            axes[1].scatter(x2, y2, s=18, color=color, label=label)

    axes[0].set_title(f"kx cut at ky={ky_cut:.6f}")
    axes[1].set_title(f"ky cut at kx={kx_cut:.6f}")
    axes[0].set_xlabel(r"$k_x$ (tpiba)")
    axes[1].set_xlabel(r"$k_y$ (tpiba)")
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylabel(ylabel)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    if fermi is not None:
        axes[0].axhline(0.0, color="gray", lw=1.0, ls="--")
        axes[1].axhline(0.0, color="gray", lw=1.0, ls="--")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)

    proj_stem = os.path.splitext(os.path.basename(args.proj_out))[0]
    save_path = args.savefig if args.savefig else f"{proj_stem}_pi_bands.png"
    fig.suptitle(
        f"pi bands from {os.path.basename(args.proj_out)} | "
        f"pi projector: l={args.pi_l}, m={args.pi_m}, "
        f"selection={args.selection_mode}, ranks={rank_ids}\n"
        f"inferred min-gap@({min_gap_info['kx']:.6f}, {min_gap_info['ky']:.6f}), "
        f"gap={min_gap_info['gap']:.6f} eV, pair={min_gap_info['pair']}",
        fontsize=10,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"Saved plot: {save_path}")
    print(f"Info: parsed_records={len(records)}, pi_states={sorted(pi_state_ids)}")
    print(f"Info: selection_mode={args.selection_mode}, selected_ranks={rank_ids}")
    for rank in rank_ids:
        print(f"  rank {rank}: mean_pi_weight={rank_stats[rank]:.6f}")


if __name__ == "__main__":
    main()
