import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

HARTREE_TO_EV = 27.2114


def parse_qe_xml(xml_path):
    """Parse k-points and eigenvalues from QE data-file-schema.xml."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    k_points = []
    eigenvalues_ev = []

    for ks in root.findall(".//ks_energies"):
        kpt_node = ks.find("k_point")
        eig_node = ks.find("eigenvalues")
        if kpt_node is None or eig_node is None:
            continue

        coords = [float(x) for x in kpt_node.text.split()]
        eigs_ev = np.array([float(x) for x in eig_node.text.split()]) * HARTREE_TO_EV
        k_points.append(coords[:2])  # keep only kx, ky
        eigenvalues_ev.append(eigs_ev)

    if not k_points or not eigenvalues_ev:
        raise ValueError("No k-point/eigenvalue data found in XML.")

    return np.array(k_points), np.array(eigenvalues_ev)


def pick_dirac_bands(eigenvalues_ev, fermi_ev):
    """Pick valence/conduction band indices nearest to the Fermi level."""
    below_fermi = eigenvalues_ev <= fermi_ev
    if not np.any(below_fermi):
        raise ValueError("No occupied states below Fermi level were found.")
    if np.all(below_fermi):
        raise ValueError("All bands are below Fermi level. Conduction band not found.")

    # At each k-point, highest occupied index; use the most frequent index globally.
    valence_indices = np.where(
        below_fermi,
        np.arange(eigenvalues_ev.shape[1]),
        -1
    ).max(axis=1)
    val_idx = int(np.bincount(valence_indices).argmax())
    cond_idx = val_idx + 1

    if cond_idx >= eigenvalues_ev.shape[1]:
        raise ValueError("Could not determine a valid conduction band index.")

    return val_idx, cond_idx


def build_regular_grid(k_points, z_values, nres=None):
    """
    Try to reshape scattered k-point data to a regular (kx, ky) mesh.
    Returns KX, KY, Z if possible, otherwise returns None.
    """
    kx = k_points[:, 0]
    ky = k_points[:, 1]
    unique_kx = np.unique(np.round(kx, 12))
    unique_ky = np.unique(np.round(ky, 12))

    if nres is not None and (len(unique_kx) != nres or len(unique_ky) != nres):
        return None

    if len(unique_kx) * len(unique_ky) != len(k_points):
        return None

    # Sort so reshape is deterministic and independent of QE output ordering.
    order = np.lexsort((ky, kx))
    k_sorted = k_points[order]
    z_sorted = z_values[order]

    nx = len(unique_kx)
    ny = len(unique_ky)
    KX = k_sorted[:, 0].reshape(nx, ny)
    KY = k_sorted[:, 1].reshape(nx, ny)
    Z = z_sorted.reshape(nx, ny)
    return KX, KY, Z

def main():
    parser = argparse.ArgumentParser(description="Plot 3D Dirac Cone from QE XML")
    parser.add_argument("--prefix", type=str, required=True, help="QE run prefix (e.g., graphene0.0zoom)")
    parser.add_argument("--fermi", type=float, required=True, help="Fermi energy in eV (from SCF run)")
    parser.add_argument("--nres", type=int, default=None, help="Expected regular grid resolution (optional)")
    parser.add_argument("--outdir", type=str, default="./tmp", help="Output directory")
    parser.add_argument(
        "--view",
        choices=["oblique", "side-kx", "side-ky", "top", "custom"],
        default="oblique",
        help="Camera preset. Use --view custom with --elev/--azim for manual control."
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Continuous up/down camera angle in degrees. 0=side view, 90=top view."
    )
    parser.add_argument("--elev", type=float, default=None, help="Camera elevation angle in degrees (up/down tilt)")
    parser.add_argument("--azim", type=float, default=None, help="Camera azimuth angle in degrees (rotation around z)")
    args = parser.parse_args()

    xml_path = os.path.join(args.outdir, f'{args.prefix}.save', 'data-file-schema.xml')

    if not os.path.exists(xml_path):
        print(f"Error: Could not find XML file at {xml_path}")
        return

    try:
        k_points, eigenvalues_ev = parse_qe_xml(xml_path)
        val_idx, cond_idx = pick_dirac_bands(eigenvalues_ev, args.fermi)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Shift energies relative to Fermi level
    val_bands = eigenvalues_ev[:, val_idx] - args.fermi
    cond_bands = eigenvalues_ev[:, cond_idx] - args.fermi

    kx = k_points[:, 0]
    ky = k_points[:, 1]

    val_grid = build_regular_grid(k_points, val_bands, args.nres)
    cond_grid = build_regular_grid(k_points, cond_bands, args.nres)

    # 3. Generate the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if val_grid is not None and cond_grid is not None:
        KX, KY, Z_val = val_grid
        _, _, Z_cond = cond_grid
        ax.plot_surface(KX, KY, Z_val, cmap='Blues', alpha=0.85, edgecolor='none')
        ax.plot_surface(KX, KY, Z_cond, cmap='Reds', alpha=0.85, edgecolor='none')
    else:
        # Non-rectangular k-point sets (common for full BZ meshes) must be triangulated.
        tri = Triangulation(kx, ky)
        ax.plot_trisurf(tri, val_bands, cmap='Blues', alpha=0.85, linewidth=0.0, antialiased=True)
        ax.plot_trisurf(tri, cond_bands, cmap='Reds', alpha=0.85, linewidth=0.0, antialiased=True)
        print("Info: k-point set is not a rectangular grid; using triangulated surfaces.")

    # Formatting
    ax.set_title(f"Dirac Cone for {args.prefix} (bands {val_idx + 1}/{cond_idx + 1})")
    ax.set_xlabel(r'$k_x$ (tpiba)')
    ax.set_ylabel(r'$k_y$ (tpiba)')
    ax.set_zlabel('Energy - $E_F$ (eV)')

    z_all = np.concatenate([val_bands, cond_bands])
    zmax = np.max(np.abs(z_all))
    if zmax > 0:
        ax.set_zlim(-zmax, zmax)

    view_presets = {
        "oblique": (22.0, 40.0),
        "side-kx": (0.0, 90.0),   # look almost along +ky to inspect E vs kx
        "side-ky": (0.0, 0.0),    # look almost along +kx to inspect E vs ky
        "top": (90.0, 0.0),       # bird's-eye view in k-space plane
    }

    if args.view == "custom":
        elev = 22.0 if args.elev is None else args.elev
        azim = 40.0 if args.azim is None else args.azim
    else:
        elev, azim = view_presets[args.view]
        # Manual overrides are still allowed on top of a preset.
        if args.elev is not None:
            elev = args.elev
        if args.azim is not None:
            azim = args.azim

    # Simple continuous control: --angle is an alias for elevation.
    if args.angle is not None:
        elev = args.angle

    ax.view_init(elev=elev, azim=azim)
    print(f"Info: using camera view elev={elev:.1f}, azim={azim:.1f}")

    # Save to file
    savename = f"{args.prefix}_3dcone.png"
    plt.savefig(savename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved 3D plot to {savename}")

if __name__ == "__main__":
    main()