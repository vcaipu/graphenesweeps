import argparse
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

HARTREE_TO_EV = 27.2114


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Debug plot: scatter raw points for selected band indices "
            "on kx and ky cuts through the Dirac-center region."
        )
    )
    parser.add_argument("--prefix", required=True, help="QE prefix (example: graphene24.0upper)")
    parser.add_argument(
        "--outdir",
        default="tmp",
        help="QE outdir containing PREFIX.save/data-file-schema.xml (default: tmp).",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default="4,5,6",
        help="1-based band indices to plot, comma-separated (default: 4,5,6).",
    )
    parser.add_argument(
        "--fermi",
        type=float,
        default=None,
        help="Optional Fermi level override in eV.",
    )
    parser.add_argument(
        "--ky",
        type=float,
        default=None,
        help="Optional ky target for kx cut. If omitted, uses Dirac-center ky.",
    )
    parser.add_argument(
        "--kx",
        type=float,
        default=None,
        help="Optional kx target for ky cut. If omitted, uses Dirac-center kx.",
    )
    parser.add_argument(
        "--savefig",
        type=str,
        default=None,
        help="Output image path (default: <prefix>_bands_debug.png).",
    )
    return parser.parse_args()


def parse_band_list(bands_text):
    try:
        bands_1based = [int(x.strip()) for x in bands_text.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --bands value '{bands_text}': {exc}") from exc
    if not bands_1based:
        raise ValueError("No valid band indices parsed from --bands.")
    if any(b <= 0 for b in bands_1based):
        raise ValueError("Band indices in --bands must be positive (1-based).")
    return bands_1based


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
        raise ValueError("No valid ks_energies entries in XML.")

    bands = np.array(eig_rows, dtype=float)
    if bands.ndim != 2:
        raise ValueError("Inconsistent eigenvalue table shape.")

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
        raise ValueError("Cannot infer nocc without nelec or fermi.")
    occ_counts = np.sum(bands_ev <= (fermi_ev + 1e-6), axis=1)
    nocc = int(np.median(occ_counts))
    if not (1 <= nocc < nbnd):
        raise ValueError("Could not infer occupied-band count.")
    return nocc


def nearest_cut_indices(k_points, target_value, along="x"):
    if along == "x":
        # kx cut: fix ky
        uniq = np.unique(np.round(k_points[:, 1], 12))
        cut_val = float(uniq[np.argmin(np.abs(uniq - target_value))])
        mask = np.isclose(k_points[:, 1], cut_val, atol=1e-10)
        axis_vals = k_points[mask, 0]
    else:
        # ky cut: fix kx
        uniq = np.unique(np.round(k_points[:, 0], 12))
        cut_val = float(uniq[np.argmin(np.abs(uniq - target_value))])
        mask = np.isclose(k_points[:, 0], cut_val, atol=1e-10)
        axis_vals = k_points[mask, 1]

    idx = np.where(mask)[0]
    order = np.argsort(axis_vals)
    return idx[order], axis_vals[order], cut_val


def main():
    args = parse_args()
    xml_path = os.path.join(args.outdir, f"{args.prefix}.save", "data-file-schema.xml")
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    try:
        bands_1based = parse_band_list(args.bands)
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

    nbnd = bands_ev.shape[1]
    bands_0based = [b - 1 for b in bands_1based]
    if any(b < 0 or b >= nbnd for b in bands_0based):
        print(f"Error: --bands {bands_1based} out of range for nbnd={nbnd}.")
        return

    # Dirac-center estimate from min direct gap between nocc and nocc+1.
    direct_gap = bands_ev[:, nocc] - bands_ev[:, nocc - 1]
    i0 = int(np.argmin(direct_gap))
    kx0 = float(k_points[i0, 0])
    ky0 = float(k_points[i0, 1])

    ky_target = ky0 if args.ky is None else args.ky
    kx_target = kx0 if args.kx is None else args.kx

    x_idx, kx_vals, ky_cut = nearest_cut_indices(k_points, ky_target, along="x")
    y_idx, ky_vals, kx_cut = nearest_cut_indices(k_points, kx_target, along="y")
    if x_idx.size < 2 or y_idx.size < 2:
        print("Error: Could not form requested cuts (too few points).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange", "tab:brown"]

    for j, (b0, b1) in enumerate(zip(bands_0based, bands_1based)):
        color = colors[j % len(colors)]
        ex = bands_ev[x_idx, b0] - fermi_ev
        ey = bands_ev[y_idx, b0] - fermi_ev
        axes[0].scatter(kx_vals, ex, s=16, color=color, label=f"Band {b1}")
        axes[1].scatter(ky_vals, ey, s=16, color=color, label=f"Band {b1}")

    axes[0].axhline(0.0, color="gray", lw=1.0, ls="--")
    axes[1].axhline(0.0, color="gray", lw=1.0, ls="--")
    axes[0].set_title(f"kx cut at ky={ky_cut:.6f}")
    axes[1].set_title(f"ky cut at kx={kx_cut:.6f}")
    axes[0].set_xlabel(r"$k_x$ (tpiba)")
    axes[1].set_xlabel(r"$k_y$ (tpiba)")
    axes[0].set_ylabel(r"$E - E_F$ (eV)")
    axes[1].set_ylabel(r"$E - E_F$ (eV)")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")

    fig.suptitle(
        f"{args.prefix} | raw band-point debug | bands={bands_1based} | "
        f"Dirac center~({kx0:.6f}, {ky0:.6f}) | nocc={nocc}",
        fontsize=11,
    )

    save_path = args.savefig if args.savefig is not None else f"{args.prefix}_bands_debug.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved debug band-point plot: {save_path}")
    print(
        f"Info: nbnd={nbnd}, bands={bands_1based}, nocc={nocc}, "
        f"dirac_center=({kx0:.8f},{ky0:.8f}), ky_cut={ky_cut:.8f}, kx_cut={kx_cut:.8f}"
    )


if __name__ == "__main__":
    main()
