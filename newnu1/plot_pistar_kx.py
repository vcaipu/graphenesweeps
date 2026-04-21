import argparse
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

HARTREE_TO_EV = 27.2114


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot only the pi* branch along a ky=const cut (default ky=0) and fit "
            "its kx dispersion near the Dirac/merging point."
        )
    )
    parser.add_argument("--prefix", required=True, help="QE prefix (example: graphene26.5upper)")
    parser.add_argument(
        "--outdir",
        default="tmp",
        help="QE outdir containing PREFIX.save/data-file-schema.xml (default: tmp).",
    )
    parser.add_argument(
        "--ky",
        type=float,
        default=None,
        help="Optional ky cut value (tpiba). If omitted, uses Dirac-center ky.",
    )
    parser.add_argument(
        "--kx",
        type=float,
        default=None,
        help="Optional kx cut value for ky-direction fit. If omitted, uses Dirac-center kx.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.08,
        help="Half-width in kx around center for local fitting (tpiba), default 0.08.",
    )
    parser.add_argument(
        "--fermi",
        type=float,
        default=None,
        help="Optional Fermi level override in eV.",
    )
    parser.add_argument(
        "--savefig",
        type=str,
        default=None,
        help="Output PNG path (default: <prefix>_pistar_kx.png).",
    )
    return parser.parse_args()


def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    k_points = []
    eig_rows = []
    for ks in root.findall(".//ks_energies"):
        kpt_node = ks.find("k_point")
        eig_node = ks.find("eigenvalues")
        if kpt_node is None or eig_node is None or kpt_node.text is None or eig_node.text is None:
            continue
        try:
            kpt = [float(x) for x in kpt_node.text.split()]
            eigs = np.array([float(x) for x in eig_node.text.split()], dtype=float) * HARTREE_TO_EV
        except ValueError:
            continue
        if len(kpt) < 2 or eigs.size < 2:
            continue
        k_points.append(kpt[:2])
        eig_rows.append(eigs)

    if not k_points:
        raise ValueError("No valid ks_energies found.")

    bands = np.array(eig_rows, dtype=float)
    if bands.ndim != 2:
        raise ValueError("Inconsistent eigenvalue array shape.")

    fermi_ev = None
    fermi_text = root.findtext(".//fermi_energy")
    if fermi_text is not None:
        try:
            fermi_ev = float(fermi_text) * HARTREE_TO_EV
        except ValueError:
            fermi_ev = None

    nelec = None
    nelec_text = root.findtext(".//nelec")
    if nelec_text is not None:
        try:
            nelec = float(nelec_text)
        except ValueError:
            nelec = None

    return np.array(k_points), bands, fermi_ev, nelec


def choose_nocc(bands_ev, nelec, fermi_ev):
    nbnd = bands_ev.shape[1]
    if nelec is not None:
        nocc = int(round(nelec / 2.0))
        if 1 <= nocc < nbnd:
            return nocc
    if fermi_ev is None:
        raise ValueError("Could not determine nocc from nelec and no Fermi level available.")
    occupied_counts = np.sum(bands_ev <= (fermi_ev + 1e-6), axis=1)
    nocc = int(np.median(occupied_counts))
    if not (1 <= nocc < nbnd):
        raise ValueError("Could not infer occupied band count.")
    return nocc


def track_branch_along_cut(
    bands_ev, cut_indices, center_pos, start_band_idx, search_radius=1, allowed_bands=None
):
    nbnd = bands_ev.shape[1]
    tracked_band = np.zeros(len(cut_indices), dtype=int)
    tracked_energy = np.zeros(len(cut_indices), dtype=float)

    k_center = int(cut_indices[center_pos])
    tracked_band[center_pos] = start_band_idx
    tracked_energy[center_pos] = bands_ev[k_center, start_band_idx]

    def walk(start, stop, step):
        for pos in range(start, stop, step):
            prev_pos = pos - step
            prev_band = int(tracked_band[prev_pos])
            prev_e = tracked_energy[prev_pos]
            k_idx = int(cut_indices[pos])
            if allowed_bands is None:
                lo = max(0, prev_band - search_radius)
                hi = min(nbnd - 1, prev_band + search_radius)
                cand_bands = np.arange(lo, hi + 1, dtype=int)
            else:
                cand_bands = np.array(
                    [b for b in allowed_bands if 0 <= b < nbnd and abs(b - prev_band) <= search_radius],
                    dtype=int,
                )
                if cand_bands.size == 0:
                    cand_bands = np.array([prev_band], dtype=int)
            cand_e = bands_ev[k_idx, cand_bands]
            best = int(np.argmin(np.abs(cand_e - prev_e)))
            tracked_band[pos] = int(cand_bands[best])
            tracked_energy[pos] = float(cand_e[best])

    walk(center_pos + 1, len(cut_indices), 1)
    walk(center_pos - 1, -1, -1)
    return tracked_energy, tracked_band


def get_cut_indices(k_points, target_value, along="x"):
    if along == "x":
        uniq = np.unique(np.round(k_points[:, 1], 12))
        cut_val = float(uniq[np.argmin(np.abs(uniq - target_value))])
        mask = np.isclose(k_points[:, 1], cut_val, atol=1e-10)
        axis_vals = k_points[mask, 0]
    else:
        uniq = np.unique(np.round(k_points[:, 0], 12))
        cut_val = float(uniq[np.argmin(np.abs(uniq - target_value))])
        mask = np.isclose(k_points[:, 0], cut_val, atol=1e-10)
        axis_vals = k_points[mask, 1]

    idx = np.where(mask)[0]
    order = np.argsort(axis_vals)
    return idx[order], float(cut_val)


def fit_direction(dq, e_rel, window):
    fit_mask = np.abs(dq) <= window
    if np.count_nonzero(fit_mask) < 5:
        raise ValueError("Not enough points in fit window; increase --window.")
    dq_fit = dq[fit_mask]
    e_fit = e_rel[fit_mask]
    p_lin = np.polyfit(np.abs(dq_fit), e_fit, 1)
    p_quad = np.polyfit(dq_fit**2, e_fit, 1)
    y_lin = p_lin[0] * np.abs(dq_fit) + p_lin[1]
    y_quad = p_quad[0] * (dq_fit**2) + p_quad[1]
    rmse_lin = float(np.sqrt(np.mean((e_fit - y_lin) ** 2)))
    rmse_quad = float(np.sqrt(np.mean((e_fit - y_quad) ** 2)))
    better = "quadratic" if rmse_quad < rmse_lin else "linear"
    return {
        "dq_fit": dq_fit,
        "p_lin": p_lin,
        "p_quad": p_quad,
        "rmse_lin": rmse_lin,
        "rmse_quad": rmse_quad,
        "better": better,
    }


def plot_panel(ax, dq, e_rel, fit, xlabel, title):
    ax.scatter(dq, e_rel, s=22, color="black", label=r"Tracked $\pi^*$ data")
    x_plot = np.linspace(np.min(fit["dq_fit"]), np.max(fit["dq_fit"]), 300)
    ax.plot(
        x_plot,
        fit["p_lin"][0] * np.abs(x_plot) + fit["p_lin"][1],
        lw=2.0,
        color="tab:blue",
        label="Linear fit",
    )
    ax.plot(
        x_plot,
        fit["p_quad"][0] * (x_plot**2) + fit["p_quad"][1],
        lw=2.0,
        color="tab:red",
        label="Quadratic fit",
    )
    ax.axvline(0.0, color="gray", lw=1.0, ls="--")
    ax.axhline(0.0, color="gray", lw=1.0, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$E_{\pi^*}-E_{\pi^*,0}$ (eV)")
    ax.set_title(f"{title}\nBetter: {fit['better']}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.02,
        (
            f"Linear RMSE={fit['rmse_lin']:.3e}\n"
            f"Quadratic RMSE={fit['rmse_quad']:.3e}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.5", "alpha": 0.9},
    )


def main():
    args = parse_args()
    xml_path = os.path.join(args.outdir, f"{args.prefix}.save", "data-file-schema.xml")
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    try:
        k_points, bands_ev, fermi_xml, nelec = read_xml(xml_path)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    fermi_ev = args.fermi if args.fermi is not None else fermi_xml
    if fermi_ev is None:
        print("Error: Could not determine Fermi level. Pass --fermi.")
        return

    try:
        nocc = choose_nocc(bands_ev, nelec, fermi_ev)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    # Global Dirac-point center from minimum direct gap in full sampled set.
    direct_gap_all = bands_ev[:, nocc] - bands_ev[:, nocc - 1]
    center_idx_global = int(np.argmin(direct_gap_all))
    kx_center = float(k_points[center_idx_global, 0])
    ky_center = float(k_points[center_idx_global, 1])

    # Cuts around the Dirac center (or user overrides), then pi* tracking constrained to low-energy subspace.
    ky_target = ky_center if args.ky is None else args.ky
    kx_target = kx_center if args.kx is None else args.kx
    x_cut_indices, ky_cut = get_cut_indices(k_points, ky_target, along="x")
    y_cut_indices, kx_cut = get_cut_indices(k_points, kx_target, along="y")
    if x_cut_indices.size < 7 or y_cut_indices.size < 7:
        print("Error: Too few points on one or both requested cuts.")
        return

    # Keep tracking in a narrow low-energy subspace to avoid sigma-band jumping.
    allowed_bands = [nocc - 1, nocc, nocc + 1]

    x_axis_vals = k_points[x_cut_indices, 0]
    x_center_pos = int(np.argmin(np.abs(x_axis_vals - kx_center)))
    e_track_x, _ = track_branch_along_cut(
        bands_ev, x_cut_indices, x_center_pos, start_band_idx=nocc, search_radius=1, allowed_bands=allowed_bands
    )
    dq_x = x_axis_vals - x_axis_vals[x_center_pos]
    e_rel_x = e_track_x - e_track_x[x_center_pos]
    fit_x = fit_direction(dq_x, e_rel_x, args.window)

    y_axis_vals = k_points[y_cut_indices, 1]
    y_center_pos = int(np.argmin(np.abs(y_axis_vals - ky_center)))
    e_track_y, _ = track_branch_along_cut(
        bands_ev, y_cut_indices, y_center_pos, start_band_idx=nocc, search_radius=1, allowed_bands=allowed_bands
    )
    dq_y = y_axis_vals - y_axis_vals[y_center_pos]
    e_rel_y = e_track_y - e_track_y[y_center_pos]
    fit_y = fit_direction(dq_y, e_rel_y, args.window)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), constrained_layout=True)
    plot_panel(
        axes[0],
        dq_x,
        e_rel_x,
        fit_x,
        xlabel=r"$\Delta k_x$ (tpiba)",
        title=f"$\\pi^*$ along kx at ky={ky_cut:.5f}",
    )
    plot_panel(
        axes[1],
        dq_y,
        e_rel_y,
        fit_y,
        xlabel=r"$\Delta k_y$ (tpiba)",
        title=f"$\\pi^*$ along ky at kx={kx_cut:.5f}",
    )
    fig.suptitle(
        f"{args.prefix} | Dirac center=({kx_center:.6f}, {ky_center:.6f}) | "
        f"bands constrained to {nocc}/{nocc+1}/{nocc+2} | window={args.window:.4f}",
        fontsize=11,
    )

    save_path = args.savefig if args.savefig is not None else f"{args.prefix}_pistar_kx.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved pi* directional-cut plot: {save_path}")
    print(
        f"Info: nocc={nocc} (bands {nocc}/{nocc+1}), center=({kx_center:.8f},{ky_center:.8f}), "
        f"ky_cut={ky_cut:.8f}, kx_cut={kx_cut:.8f}, "
        f"x_rmse_lin={fit_x['rmse_lin']:.6e}, x_rmse_quad={fit_x['rmse_quad']:.6e}, "
        f"y_rmse_lin={fit_y['rmse_lin']:.6e}, y_rmse_quad={fit_y['rmse_quad']:.6e}"
    )


if __name__ == "__main__":
    main()
