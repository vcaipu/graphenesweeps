import argparse
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.tri import Triangulation

HARTREE_TO_EV = 27.2114


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a 2D heatmap of the direct gap Ec(k)-Ev(k) for every k-point "
            "contained in a QE data-file-schema.xml file."
        )
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="QE run prefix (example: graphene25.1upper)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="tmp",
        help="QE outdir where PREFIX.save/data-file-schema.xml lives (default: tmp).",
    )
    parser.add_argument(
        "--nocc",
        type=int,
        default=None,
        help="Occupied-band count. If omitted, inferred from nelec or Fermi level.",
    )
    parser.add_argument(
        "--fermi",
        type=float,
        default=None,
        help="Fermi level in eV (optional override used only when nocc is not provided).",
    )
    parser.add_argument(
        "--axes",
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Which k components to display in 2D (default: xy).",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=200,
        help="Number of contour levels for triangulated plotting (default: 200).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap name (default: coolwarm).",
    )
    parser.add_argument("--vmin", type=float, default=None, help="Lower color limit in eV.")
    parser.add_argument("--vmax", type=float, default=None, help="Upper color limit in eV.")
    parser.add_argument(
        "--hide-min",
        action="store_true",
        help="Hide minimum-point markers/labels (shown by default).",
    )
    parser.add_argument(
        "--min-mode",
        choices=["both", "global"],
        default="both",
        help="Minima to mark: both (ky>0 and ky<0) or global (default: both).",
    )
    parser.add_argument(
        "--label-gap",
        action="store_true",
        help="Include gap values in minimum-point labels (off by default).",
    )
    return parser.parse_args()


def read_qe_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    k_points = []
    eigenvalues_ev = []
    for ks in root.findall(".//ks_energies"):
        kpt_node = ks.find("k_point")
        eig_node = ks.find("eigenvalues")
        if kpt_node is None or eig_node is None or eig_node.text is None or kpt_node.text is None:
            continue

        try:
            kpt = [float(x) for x in kpt_node.text.split()]
            eigs_ev = np.array([float(x) for x in eig_node.text.split()]) * HARTREE_TO_EV
        except ValueError:
            continue

        if len(kpt) < 3 or eigs_ev.size < 2:
            continue

        k_points.append(kpt[:3])
        eigenvalues_ev.append(eigs_ev)

    if not k_points:
        raise ValueError("No valid ks_energies entries found in XML.")

    bands = np.array(eigenvalues_ev)
    if bands.ndim != 2:
        raise ValueError("Parsed eigenvalue table has inconsistent band counts across k-points.")

    return root, np.array(k_points), bands


def parse_nelec(root):
    text = root.findtext(".//nelec")
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_fermi_from_xml_ev(root):
    text = root.findtext(".//fermi_energy")
    if text is None:
        return None
    try:
        return float(text) * HARTREE_TO_EV
    except ValueError:
        return None


def infer_nocc_from_fermi(eigenvalues_ev, fermi_ev, tol_ev=1e-6):
    if fermi_ev is None:
        return None
    occ_counts = np.sum(eigenvalues_ev <= (fermi_ev + tol_ev), axis=1)
    if occ_counts.size == 0:
        return None
    nocc = int(np.median(occ_counts))
    if nocc <= 0 or nocc >= eigenvalues_ev.shape[1]:
        return None
    return nocc


def choose_nocc(root, eigenvalues_ev, nocc_cli=None, fermi_cli_ev=None):
    nbnd = eigenvalues_ev.shape[1]

    if nocc_cli is not None:
        nocc = int(nocc_cli)
        source = "user --nocc"
    else:
        nelec = parse_nelec(root)
        if nelec is not None:
            nocc = int(round(nelec / 2.0))
            source = "xml nelec/2"
        else:
            nocc = None
            source = None

        if nocc is None or nocc <= 0 or nocc >= nbnd:
            fermi_ev = fermi_cli_ev if fermi_cli_ev is not None else parse_fermi_from_xml_ev(root)
            nocc = infer_nocc_from_fermi(eigenvalues_ev, fermi_ev)
            source = "Fermi occupancy inference"

    if nocc is None or nocc <= 0 or nocc >= nbnd:
        raise ValueError(
            "Could not determine occupied-band count. Pass --nocc explicitly (1 <= nocc < nbnd)."
        )

    return nocc, source


def pick_axes(k_points, axes):
    axis_map = {
        "xy": (0, 1, r"$k_x$ (tpiba)", r"$k_y$ (tpiba)"),
        "xz": (0, 2, r"$k_x$ (tpiba)", r"$k_z$ (tpiba)"),
        "yz": (1, 2, r"$k_y$ (tpiba)", r"$k_z$ (tpiba)"),
    }
    i, j, xlabel, ylabel = axis_map[axes]
    return k_points[:, i], k_points[:, j], xlabel, ylabel


def build_regular_grid(x, y, z):
    x_rounded = np.round(x, 12)
    y_rounded = np.round(y, 12)
    unique_x = np.unique(x_rounded)
    unique_y = np.unique(y_rounded)

    nx = len(unique_x)
    ny = len(unique_y)
    if nx * ny != len(x):
        return None

    x_to_idx = {value: idx for idx, value in enumerate(unique_x)}
    y_to_idx = {value: idx for idx, value in enumerate(unique_y)}

    grid = np.full((nx, ny), np.nan, dtype=float)
    for xi, yi, zi in zip(x_rounded, y_rounded, z):
        ix = x_to_idx[xi]
        iy = y_to_idx[yi]
        if not np.isnan(grid[ix, iy]):
            return None
        grid[ix, iy] = zi

    if np.isnan(grid).any():
        return None

    X, Y = np.meshgrid(unique_x, unique_y, indexing="ij")
    return X, Y, grid


def main():
    args = parse_args()
    xml_path = os.path.join(args.outdir, f"{args.prefix}.save", "data-file-schema.xml")

    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    try:
        root, k_points, eigenvalues_ev = read_qe_xml(xml_path)
        nocc, nocc_source = choose_nocc(root, eigenvalues_ev, args.nocc, args.fermi)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    val_idx = nocc - 1
    cond_idx = nocc
    valence_ev = eigenvalues_ev[:, val_idx]
    conduction_ev = eigenvalues_ev[:, cond_idx]
    direct_gap_ev = conduction_ev - valence_ev

    x, y, xlabel, ylabel = pick_axes(k_points, args.axes)
    zmin = float(np.min(direct_gap_ev))
    zmax = float(np.max(direct_gap_ev))

    if args.vmin is not None and args.vmax is not None and args.vmin >= args.vmax:
        print("Error: --vmin must be smaller than --vmax.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    grid_data = build_regular_grid(x, y, direct_gap_ev)

    norm = None
    if args.vmin is None and args.vmax is None and zmin < 0.0 < zmax:
        norm = TwoSlopeNorm(vcenter=0.0, vmin=zmin, vmax=zmax)

    if grid_data is not None:
        X, Y, Z = grid_data
        mesh = ax.pcolormesh(
            X, Y, Z, shading="auto", cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, norm=norm
        )
    else:
        tri = Triangulation(x, y)
        mesh = ax.tricontourf(
            tri, direct_gap_ev, levels=args.levels, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, norm=norm
        )
        print("Info: k-point set is not rectangular in chosen axes; using triangulated contour heatmap.")

    selected_minima = []
    if args.min_mode == "global":
        min_idx = int(np.argmin(direct_gap_ev))
        selected_minima.append(("global", min_idx))
    else:
        ky = k_points[:, 1]
        upper_mask = ky > 0.0
        lower_mask = ky < 0.0
        if not np.any(upper_mask) or not np.any(lower_mask):
            print("Error: Could not find both ky>0 and ky<0 k-point regions.")
            return
        upper_indices = np.where(upper_mask)[0]
        lower_indices = np.where(lower_mask)[0]
        upper_min_idx = int(upper_indices[np.argmin(direct_gap_ev[upper_mask])])
        lower_min_idx = int(lower_indices[np.argmin(direct_gap_ev[lower_mask])])
        selected_minima.extend([("upper", upper_min_idx), ("lower", lower_min_idx)])

    if not args.hide_min:
        for label, idx in selected_minima:
            min_gap = float(direct_gap_ev[idx])
            min_x = float(x[idx])
            min_y = float(y[idx])
            min_kx = float(k_points[idx, 0])
            min_ky = float(k_points[idx, 1])
            xoff = -120 if min_x > 0 else 8
            yoff = -14 if min_y > 0 else 8
            label_text = f"{label} (kx, ky)=({min_kx:.5f}, {min_ky:.5f})"
            if args.label_gap:
                label_text = f"{label} min gap={min_gap:.6f} eV\n(kx, ky)=({min_kx:.5f}, {min_ky:.5f})"
            ax.scatter([min_x], [min_y], s=120, color="black", marker="x", linewidths=2.2)
            ax.annotate(
                label_text,
                (min_x, min_y),
                xytext=(xoff, yoff),
                textcoords="offset points",
                color="black",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.2", "alpha": 0.8},
            )

    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(r"$E_c(k)-E_v(k)$ (eV)")

    ax.set_title(
        f"Direct Gap Heatmap: {args.prefix}\n"
        f"(valence band #{val_idx + 1}, conduction band #{cond_idx + 1}; nocc source: {nocc_source})"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")

    outname = f"{args.prefix}_heatmap_{args.axes}.png"
    plt.tight_layout()
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    print(f"Success: wrote heatmap to {outname}")
    minima_info = ", ".join([f"{label}_min_gap={float(direct_gap_ev[idx]):.8f} eV" for label, idx in selected_minima])
    print(f"Info: nk={len(direct_gap_ev)}, nbnd={eigenvalues_ev.shape[1]}, {minima_info}")


if __name__ == "__main__":
    main()
