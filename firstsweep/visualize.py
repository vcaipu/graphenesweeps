import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot 3D Dirac Cone from QE XML")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix of the zoom run (e.g., graphene12.0zoom)")
    parser.add_argument("--fermi", type=float, required=True, help="Fermi energy in eV (from SCF run)")
    parser.add_argument("--nres", type=int, default=40, help="Grid resolution used in the zoom (e.g., 40)")
    parser.add_argument("--outdir", type=str, default="./tmp", help="Output directory")
    args = parser.parse_args()

    xml_path = os.path.join(args.outdir, f'{args.prefix}.save', 'data-file-schema.xml')

    if not os.path.exists(xml_path):
        print(f"Error: Could not find XML file at {xml_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. Parse k-points
    k_points = []
    for kpt in root.findall(".//k_point"):
        coords = [float(x) for x in kpt.text.split()]
        k_points.append(coords[:2])  # We only need kx, ky
    
    k_points = np.array(k_points)

    # 2. Parse valence and conduction bands
    val_bands = []
    cond_bands = []
    for energies in root.findall(".//ks_energies"):
        eig_str = energies.find("eigenvalues").text
        # Convert Hartree to eV
        eigs = np.array([float(x) for x in eig_str.split()]) * 27.2114
        val_bands.append(eigs[3])  # Band 4 (Index 3)
        cond_bands.append(eigs[4])  # Band 5 (Index 4)

    # Shift energies relative to Fermi level
    val_bands = np.array(val_bands) - args.fermi
    cond_bands = np.array(cond_bands) - args.fermi


    # 3. Handle Appended XMLs and Reshape
    total_expected = args.nres * args.nres
    
    if len(k_points) > total_expected:
        print(f"Warning: Found {len(k_points)} points. Taking the last {total_expected} from the newest run.")
        k_points = k_points[-total_expected:]
        val_bands = val_bands[-total_expected:]
        cond_bands = cond_bands[-total_expected:]
    elif len(k_points) < total_expected:
        print(f"Error: Found {len(k_points)} points, but expected {total_expected}.")
        return

    try:
        KX = k_points[:, 0].reshape(args.nres, args.nres)
        KY = k_points[:, 1].reshape(args.nres, args.nres)
        Z_val = val_bands.reshape(args.nres, args.nres)
        Z_cond = cond_bands.reshape(args.nres, args.nres)
    except ValueError as e:
        print(f"Reshape Error: {e}")
        return 

    # 4. Generate the Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the valence band (blue-ish) and conduction band (red-ish)
    surf1 = ax.plot_surface(KX, KY, Z_val, cmap='Blues', alpha=0.8, edgecolor='none', vmin=-1.0, vmax=0.0)
    surf2 = ax.plot_surface(KX, KY, Z_cond, cmap='Reds', alpha=0.8, edgecolor='none', vmin=0.0, vmax=1.0)

    # Formatting
    ax.set_title(f"Dirac Cone for {args.prefix}")
    ax.set_xlabel(r'$k_x$ (tpiba)')
    ax.set_ylabel(r'$k_y$ (tpiba)')
    ax.set_zlabel('Energy - $E_F$ (eV)')
    
    # Adjust viewing angle (elevation, azimuth)
    ax.view_init(elev=15, azim=45)

    # Save to file
    savename = f"{args.prefix}_3dcone.png"
    plt.savefig(savename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved 3D plot to {savename}")

if __name__ == "__main__":
    main()