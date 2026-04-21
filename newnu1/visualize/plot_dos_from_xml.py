import argparse
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

HARTREE_TO_EV = 27.2114


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot plain (unsmeared) DOS vs energy from QE XML eigenvalues."
    )
    parser.add_argument("--prefix", required=True, help="QE prefix (example: graphene20.0)")
    parser.add_argument(
        "--outdir",
        default="tmp",
        help="QE outdir where PREFIX.save/data-file-schema.xml is stored (default: tmp).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=300,
        help="Number of histogram bins for plain DOS (default: 300).",
    )
    parser.add_argument(
        "--fermi",
        type=float,
        default=None,
        help="Optional Fermi level override in eV. If omitted, read from XML.",
    )
    parser.add_argument(
        "--absolute-energy",
        action="store_true",
        help="Plot absolute energy E (eV) instead of E-Ef.",
    )
    parser.add_argument(
        "--emin",
        type=float,
        default=-0.5,
        help="Lower x-limit in plotted energy units (default: -0.5).",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=0.5,
        help="Upper x-limit in plotted energy units (default: 0.5).",
    )
    parser.add_argument(
        "--savefig",
        type=str,
        default=None,
        help="Output image path (default: <prefix>_dos.png).",
    )
    return parser.parse_args()


def read_xml_eigens(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    eig_rows = []
    for ks in root.findall(".//ks_energies"):
        eig_node = ks.find("eigenvalues")
        if eig_node is None or eig_node.text is None:
            continue
        try:
            eigs_ev = np.array([float(x) for x in eig_node.text.split()], dtype=float) * HARTREE_TO_EV
        except ValueError:
            continue
        if eigs_ev.size > 0:
            eig_rows.append(eigs_ev)

    if not eig_rows:
        raise ValueError("No valid eigenvalues found in XML.")

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

    return bands, fermi_ev


def main():
    args = parse_args()
    xml_path = os.path.join(args.outdir, f"{args.prefix}.save", "data-file-schema.xml")
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    try:
        bands, fermi_xml = read_xml_eigens(xml_path)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    fermi_ev = args.fermi if args.fermi is not None else fermi_xml
    energies = bands.ravel()

    if args.absolute_energy or fermi_ev is None:
        x_energies = energies
        xlabel = "Energy (eV)"
        ef_x = fermi_ev
    else:
        x_energies = energies - fermi_ev
        xlabel = r"$E - E_F$ (eV)"
        ef_x = 0.0

    counts, edges = np.histogram(x_energies, bins=args.bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(centers, counts, where="mid", color="tab:blue", lw=1.8, label="DOS (plain histogram)")
    if ef_x is not None:
        ax.axvline(ef_x, color="black", lw=1.2, ls="--", label=r"$E_F$")
    ax.set_xlim(args.emin, args.emax)
    ax.set_title(f"DOS: {args.prefix}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("DOS (counts/bin)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    save_path = args.savefig if args.savefig is not None else f"{args.prefix}_dos.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved DOS plot: {save_path}")
    print(
        f"Info: nk={bands.shape[0]}, nbnd={bands.shape[1]}, bins={args.bins}, "
        f"fermi={'None' if fermi_ev is None else f'{fermi_ev:.6f} eV'}"
    )


if __name__ == "__main__":
    main()
