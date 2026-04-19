# Copy everything you need (this script, pseudopotentials, templates, and checkpoints)
# scp -r ~/chm510/graphenestraintest/masterscript.py ~/chm510/graphenestraintest/checkpoints/ ~/chm510/graphenestraintest/*.UPF ~/chm510/graphenestraintest/templates/ .
# mkdir tmp

# Imports
import argparse
import math
import subprocess
import os
import re
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import time
coarse_resolution = 61
fine_resolution = 41
fine_radius = 0.04
scf_resolution = 24
outdir = "./tmp"
# pseudopotential = "C.pbe-n-kjpaw_psl.1.0.0.UPF"
pseudopotential = "C.upf"
# Helper functions
def calculate_strained_lattice(strain_pct):
    # --- Physical Constants ---

    a = 4.625923576  # vc-relaxed lattice constant (Bohrs, for LDA functional)
    # a = 4.654           # Pristine graphene lattice constant (Bohr)
    nu = 0.165          # Poisson ratio for free-standing graphene
    c = 28.346456       # Vacuum gap in z-direction (Bohr)
    
    # --- Deformation Gradient Tensor Components ---
    # epsilon is the fractional uniaxial strain
    epsilon = strain_pct / 100.0
    F_11 = 1.0 + epsilon
    F_22 = 1.0 - (nu * epsilon)

    # Armchair Strain Deformation
    # F_11 = 1.0 - (nu * epsilon)  # Poisson contraction along x
    # F_22 = 1.0 + epsilon         # Uniaxial tension along y

    # --- Affine Transformation ---
    # a1' = F * a1
    a1_x = a * F_11
    a1_y = 0.0
    a1_z = 0.0
    
    # a2' = F * a2
    a2_x = (a / 2.0) * F_11
    a2_y = (a * math.sqrt(3) / 2.0) * F_22
    a2_z = 0.0
    
    # a3' = F * a3 (Unstrained vacuum gap)
    a3_x = 0.0
    a3_y = 0.0
    a3_z = c

    return (a1_x, a1_y, a1_z), (a2_x, a2_y, a2_z), (a3_x, a3_y, a3_z)

def create_from_template(template_path, output_input_name, replacements):
    # 1. Read the template
    with open(template_path, 'r') as f:
        content = f.read()

    # 2. Perform the replacements
    # 'replacements' is a dictionary: {"{TAG}": "Actual Data"}
    # NOTE: The string .replace() is case-sensitive.
    for tag, value in replacements.items():
        content = content.replace(tag, value)
 

    # 3. Write the new executable input file
    with open(output_input_name, 'w') as f:
        f.write(content)

def get_initial_relax_guess(guess):
    c1x, c1y ,c1z = 0.000000,  0.000000,  0.000000
    c2x, c2y, c2z = 0.333333,  0.333333,  0.000000

    # read the checkpoint for the initial guess, if it does not exist, use the default values
    checkpoint_file = f"checkpoints/checkpoint{guess}.pkl"
    if not os.path.exists(checkpoint_file):
        return (c1x,c1y,c1z),(c2x,c2y,c2z)
    with open(checkpoint_file, 'rb') as f:
        print(f"Using initial guess from checkpoint file: {checkpoint_file}")
        checkpoint = pickle.load(f)
    c1x = checkpoint.get("relaxed_c1_vec")[0]
    c1y = checkpoint.get("relaxed_c1_vec")[1]
    c1z = checkpoint.get("relaxed_c1_vec")[2]
    c2x = checkpoint.get("relaxed_c2_vec")[0]
    c2y = checkpoint.get("relaxed_c2_vec")[1]
    print(f"Initial guess is: C1:({c1x}, {c1y}, {c1z}) | C2:({c2x}, {c2y}, {c2z})")

    return (c1x,c1y,c1z),(c2x,c2y,c2z)

def run_qe(inputfile,outputfile):
    # cmd = f"apptainer exec --nv --bind /scratch:/scratch --bind $(pwd):$(pwd) ~/software/quantum_espresso/quantum_espresso_qe-7.3.1.sif mpirun -np 1 pw.x -input {inputfile} > {outputfile}"

    # the export OMP_NUM_THREADS=1 is to prevent the code from using more than one thread, is VERY IMPORTANT
    mca_flags = "--mca btl self,vader --mca pml ob1"
    cmd = f"export OMP_NUM_THREADS=1; mpirun {mca_flags} --bind-to none --oversubscribe -np 8 pw.x -nk 8 -input {inputfile} > {outputfile}"
    subprocess.run(cmd, shell=True, check=True)

# def run_qe(inputfile, outputfile):
#     pw_path = "pw.x"
    
#     mca_flags = "--mca btl self,vader --mca pml ob1"
    
#     # -H localhost:16 tells OpenMPI: "Ignore Slurm. This node has 16 slots."
#     # --bind-to core: Overload-allowed is no longer needed because we 'liberated' the slots.
#     parallel_flags = "-H localhost:16 --report-bindings --bind-to core -np 8"
    
#     cmd = f"export OMP_NUM_THREADS=1; mpirun {mca_flags} {parallel_flags} {pw_path} -nk 8 -input {inputfile} > {outputfile}"
    
#     print(f"Executing: {cmd}")
#     subprocess.run(cmd, shell=True, check=True)


def parse_relaxed_atomic_positions(relax_output_path):
    """
    Parse the final ATOMIC_POSITIONS (crystal) block from a QE relax output.
    Returns six floats: (c1x, c1y, c1z, c2x, c2y, c2z).
    """
    with open(relax_output_path, 'r') as f:
        lines = f.readlines()

    final_block_start = None
    for idx, line in enumerate(lines):
        if line.strip() == "ATOMIC_POSITIONS (crystal)":
            final_block_start = idx

    if final_block_start is None:
        raise ValueError(
            f"Could not find 'ATOMIC_POSITIONS (crystal)' in {relax_output_path}"
        )

    positions = []
    for line in lines[final_block_start + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("End final coordinates"):
            break

        parts = stripped.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            positions.append((x, y, z))
        except ValueError:
            continue

    if len(positions) < 2:
        raise ValueError(
            f"Expected at least 2 atomic positions in final block of {relax_output_path}"
        )

    c1x, c1y, c1z = positions[0]
    c2x, c2y, c2z = positions[1]
    return (c1x, c1y, c1z), (c2x, c2y, c2z)

def parse_final_fermi_energy(relax_output_path):
    """
    Parse the final Fermi energy (in eV) from a QE relax output.
    Returns one float (fermi_ev).
    """
    with open(relax_output_path, 'r') as f:
        lines = f.readlines()

    fermi_ev = None
    pattern = re.compile(r"the Fermi energy is\s+([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)\s+ev")

    for line in lines:
        match = pattern.search(line)
        if match:
            fermi_ev = float(match.group(1))

    if fermi_ev is None:
        raise ValueError(f"Could not find final Fermi energy in {relax_output_path}")

    return fermi_ev

def get_dirac_cone(prefix, e_fermi):
    xml_path = os.path.join(outdir, f'{prefix}.save', 'data-file-schema.xml')
    
    if not os.path.exists(xml_path):
        print(f"Error: Could not find XML file at {xml_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. Parse k-point coordinates
    k_points = []
    for kpt in root.findall(".//k_point"):
        coords = [float(x) for x in kpt.text.split()]
        k_points.append(coords[:2]) # Keep only kx, ky
    k_points = np.array(k_points)

    # 2. Parse Band 4 (Valence) and Band 5 (Conduction)
    bands_data = []
    for energies in root.findall(".//ks_energies"):
        eig_str = energies.find("eigenvalues").text
        # Convert Hartree to eV
        eigs = np.array([float(x) for x in eig_str.split()]) * 27.2114
        bands_data.append(eigs)
    
    bands = np.array(bands_data) - e_fermi # Shape: (nkx*nky, nbnd)
    val_band = bands[:, 3] # Band 4 (Index 3)
    cond_band = bands[:, 4] # Band 5 (Index 4)

    # 3. Compute the rigorous band gap
    energy_differences = cond_band - val_band
    min_gap = np.min(energy_differences)
    min_gap_idx = np.argmin(energy_differences)
    kx_dirac, ky_dirac = k_points[min_gap_idx]

    return min_gap, kx_dirac, ky_dirac

def generate_sniper_grid(center_kx, center_ky, spread, n_points):
    # Generate the linear ranges
    kx_range = np.linspace(center_kx - spread, center_kx + spread, n_points)
    ky_range = np.linspace(center_ky - spread, center_ky + spread, n_points)
    
    total_k = n_points**2

    lines = [
        "K_POINTS {tpiba}",
        f"{total_k}",
    ]

    for kx in kx_range:
        for ky in ky_range:
            # We use weight 1.0 for bands calculations
            lines.append(f"  {kx:10.6f}  {ky:10.6f}  0.000000  1.0")

    # Match print() behavior exactly: one trailing newline at end.
    return "\n".join(lines) + "\n"

'''
Code begins here
'''

# Start timer
start_time = time.time()

# Parse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--strain", type=float, required=True, help="The value of the strain")
parser.add_argument("--guess", type=float, help="The strain of the initial guess (optional)")

args = parser.parse_args()
strain = args.strain
strainString = f"{strain:.1f}"

if args.guess is not None:
    guess = args.guess
    guessString = f"{guess:.1f}"
else:
    guess = None
    guessString = None  # Or provide a default if required elsewhere

prefix = f"graphene{strainString}"
zoomprefix = f"graphene{strainString}zoom"

print(f"Prefix is: {prefix}")


def write_checkpoint(checkpoint):
    """
    Merge new key/value pairs into the existing checkpoint file.
    New keys are appended; existing keys are overwritten by new values.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError("write_checkpoint expects a checkpoint dict")

    existing_checkpoint = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            loaded_checkpoint = pickle.load(f)
        if isinstance(loaded_checkpoint, dict):
            existing_checkpoint = loaded_checkpoint

    merged_checkpoint = dict(existing_checkpoint)
    merged_checkpoint.update(checkpoint)

    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(merged_checkpoint, f)
    print(f"Checkpoint written to: {checkpoint_file}")

# Open a checkpoint file, as a pickle file
checkpoint_file = f"checkpoints/checkpoint{strainString}.pkl"
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"Checkpoint file found: {checkpoint_file}")
else:
    checkpoint = {}
    write_checkpoint(checkpoint)

# Step 1: Get the Lattice vectors 
# Get strained lattice vectors

print(f"\n\n\n Step 1: Relax")
a1_vec, a2_vec, a3_vec = calculate_strained_lattice(strain)
c1_vec, c2_vec = get_initial_relax_guess(guessString)
if checkpoint.get("relaxed_c1_vec") is None or checkpoint.get("relaxed_c2_vec") is None or checkpoint.get("relax_fermi_energy") is None:
    relax_in_replacements = {
        "A1X": f"{a1_vec[0]:.6f}",
        "A1Y": f"{a1_vec[1]:.6f}",
        "A1Z": f"{a1_vec[2]:.6f}",
        "A2X": f"{a2_vec[0]:.6f}",
        "A2Y": f"{a2_vec[1]:.6f}",
        "A2Z": f"{a2_vec[2]:.6f}",
        "A3X": f"{a3_vec[0]:.6f}",
        "A3Y": f"{a3_vec[1]:.6f}",
        "A3Z": f"{a3_vec[2]:.6f}",
        "C1X": f"{c1_vec[0]:.6f}",
        "C1Y": f"{c1_vec[1]:.6f}",
        "C1Z": f"{c1_vec[2]:.6f}",
        "C2X": f"{c2_vec[0]:.6f}",
        "C2Y": f"{c2_vec[1]:.6f}",
        "C2Z": f"{c2_vec[2]:.6f}",
        "NRES": f"{scf_resolution}",
        "OUTDIR": outdir,
        "PREFIX": prefix,
        "PSEUDOPOTENTIAL": pseudopotential
    }
    inputfile1 = f"relax{strainString}.in"
    outputfile1 = f"relax{strainString}.out"
    create_from_template("templates/relax.in", inputfile1, relax_in_replacements)
    print(f"Created {inputfile1} | Running RELAX | Check: tail -n 20 {outputfile1}")

    ## THE RUN
    run_qe(inputfile1,outputfile1)

    # RELAXED COORDS
    relaxed_c1_vec, relaxed_c2_vec = parse_relaxed_atomic_positions(outputfile1)
    relax_fermi_energy = parse_final_fermi_energy(outputfile1)

    print(f"Finished Running a Relax")
    print(f"Coordinates are: C1:{ relaxed_c1_vec } | C2:{ relaxed_c2_vec } | Fermi Energy: { relax_fermi_energy } eV")
    write_checkpoint({
        "relaxed_c1_vec": relaxed_c1_vec,
        "relaxed_c2_vec": relaxed_c2_vec,
        "relax_fermi_energy": relax_fermi_energy
    })
else:
    print(f"RELAX Results found in Checkpoint file")
    relaxed_c1_vec = checkpoint.get("relaxed_c1_vec")
    relaxed_c2_vec = checkpoint.get("relaxed_c2_vec")
    relax_fermi_energy = checkpoint.get("relax_fermi_energy")
    print(f"C1: {relaxed_c1_vec} | C2:{relaxed_c2_vec} | Fermi Energy:{relax_fermi_energy} eV")

# Step 2: Get the Band Structure
print(f"\n\n\n Step 2: Coarse Bands")
if checkpoint.get("kx_dirac") is None or checkpoint.get("ky_dirac") is None:
    band_in_replacements = {
        "PREFIX": prefix,
        "OUTDIR": outdir,
        "C1X": f"{relaxed_c1_vec[0]:.6f}",
        "C1Y": f"{relaxed_c1_vec[1]:.6f}",
        "C1Z": f"{relaxed_c1_vec[2]:.6f}",
        "C2X": f"{relaxed_c2_vec[0]:.6f}",
        "C2Y": f"{relaxed_c2_vec[1]:.6f}",
        "C2Z": f"{relaxed_c2_vec[2]:.6f}",
        "A1X": f"{a1_vec[0]:.6f}",
        "A1Y": f"{a1_vec[1]:.6f}",
        "A1Z": f"{a1_vec[2]:.6f}",
        "A2X": f"{a2_vec[0]:.6f}",
        "A2Y": f"{a2_vec[1]:.6f}",
        "A2Z": f"{a2_vec[2]:.6f}",
        "A3X": f"{a3_vec[0]:.6f}",
        "A3Y": f"{a3_vec[1]:.6f}",
        "A3Z": f"{a3_vec[2]:.6f}",
        "NRES": f"{coarse_resolution}",
        "PSEUDOPOTENTIAL": pseudopotential
    }
    inputfile2 = f"bands{strainString}.in"
    outputfile2 = f"bands{strainString}.out"
    create_from_template("templates/bands.in", inputfile2, band_in_replacements)
    print(f"Created {inputfile2} | Running BANDS | Check: tail -n 20 {outputfile2}")

    ## THE RUN
    run_qe(inputfile2,outputfile2)

    # BANDS
    min_gap, kx_dirac, ky_dirac = get_dirac_cone(prefix, relax_fermi_energy)
    print(f"Finished Running a BANDS")
    print(f"Minimum gap is: {min_gap} eV")
    print(f"Dirac point is at: ({kx_dirac}, {ky_dirac})")
    write_checkpoint({
        "kx_dirac": kx_dirac,
        "ky_dirac": ky_dirac
    })
else:
    print(f"BANDS Results found in Checkpoint file")
    kx_dirac = checkpoint.get("kx_dirac")
    ky_dirac = checkpoint.get("ky_dirac")

# Step 3: Run an SCF
print(f"\n\n\n Step 3: SCF")
if checkpoint.get("e_fermi") is None:
    scf_in_replacements = {
        "PREFIX": zoomprefix,
        "OUTDIR": outdir,
        "C1X": f"{relaxed_c1_vec[0]:.6f}",
        "C1Y": f"{relaxed_c1_vec[1]:.6f}",
        "C1Z": f"{relaxed_c1_vec[2]:.6f}",
        "C2X": f"{relaxed_c2_vec[0]:.6f}",
        "C2Y": f"{relaxed_c2_vec[1]:.6f}",
        "C2Z": f"{relaxed_c2_vec[2]:.6f}",
        "A1X": f"{a1_vec[0]:.6f}",
        "A1Y": f"{a1_vec[1]:.6f}",
        "A1Z": f"{a1_vec[2]:.6f}", 
        "A2X": f"{a2_vec[0]:.6f}",
        "A2Y": f"{a2_vec[1]:.6f}",
        "A2Z": f"{a2_vec[2]:.6f}",
        "A3X": f"{a3_vec[0]:.6f}",
        "A3Y": f"{a3_vec[1]:.6f}",
        "A3Z": f"{a3_vec[2]:.6f}",
        "NRES": f"{scf_resolution}",
        "PSEUDOPOTENTIAL": pseudopotential
    }
    inputfile3 = f"scf{strainString}.in"
    outputfile3 = f"scf{strainString}.out"
    create_from_template("templates/scf.in", inputfile3, scf_in_replacements)
    print(f"Created {inputfile3} | Running SCF | Check: tail -n 20 {outputfile3}")

    ## THE RUN
    run_qe(inputfile3,outputfile3)

    # SCF
    e_fermi = parse_final_fermi_energy(outputfile3)
    print(f"Finished Running a SCF")
    print(f"Fermi Energy is: {e_fermi} eV")
    write_checkpoint({
        "e_fermi": e_fermi
    })
else:
    print(f"SCF Results found in Checkpoint file")
    e_fermi = checkpoint.get("e_fermi")

# Step 4: Run a Zoomed in Bands
print(f"\n\n\n Step 4: Zoomed in Bands")
if checkpoint.get("zoom_min_gap") is None:
    zoom_in_replacements = {
        "PREFIX": zoomprefix,
        "OUTDIR": outdir,
        "C1X": f"{relaxed_c1_vec[0]:.6f}",
        "C1Y": f"{relaxed_c1_vec[1]:.6f}",
        "C1Z": f"{relaxed_c1_vec[2]:.6f}",
        "C2X": f"{relaxed_c2_vec[0]:.6f}",
        "C2Y": f"{relaxed_c2_vec[1]:.6f}",
        "C2Z": f"{relaxed_c2_vec[2]:.6f}",
        "A1X": f"{a1_vec[0]:.6f}",
        "A1Y": f"{a1_vec[1]:.6f}",
        "A1Z": f"{a1_vec[2]:.6f}",
        "A2X": f"{a2_vec[0]:.6f}",
        "A2Y": f"{a2_vec[1]:.6f}",
        "A2Z": f"{a2_vec[2]:.6f}",
        "A3X": f"{a3_vec[0]:.6f}",
        "A3Y": f"{a3_vec[1]:.6f}",
        "A3Z": f"{a3_vec[2]:.6f}",
        "NRES": f"{fine_resolution}",
        "PSEUDOPOTENTIAL": pseudopotential
    }
    inputfile4 = f"bands{strainString}zoom.in"
    outputfile4 = f"bands{strainString}zoom.out"
    create_from_template("templates/bands.in", inputfile4, zoom_in_replacements)
    append_string = generate_sniper_grid(kx_dirac, ky_dirac, fine_radius, fine_resolution)

    # Delete the last two lines of inputfile4, and append the append_string
    with open(inputfile4, 'r') as f:
        lines = f.readlines()
    with open(inputfile4, 'w') as f:
        for line in lines[:-2]:
            f.write(line)
        f.write(append_string)
    print(f"Created {inputfile4} | Running BANDS ZOOM | Check: tail -n 20 {outputfile4}")

    ## THE RUN
    run_qe(inputfile4,outputfile4)

    # BANDS
    zoom_min_gap, zoom_kx_dirac, zoom_ky_dirac = get_dirac_cone(zoomprefix, e_fermi)
    print(f"Finished Running a BANDS")
    print(f"Minimum gap is: {zoom_min_gap} eV")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    write_checkpoint({
        "zoom_kx_dirac": zoom_kx_dirac,
        "zoom_ky_dirac": zoom_ky_dirac,
        "zoom_min_gap": zoom_min_gap,
        "time_taken": end_time - start_time
    })
else:
    print(f"BANDS ZOOM Results found in Checkpoint file")
    zoom_min_gap = checkpoint.get("zoom_min_gap")
    zoom_kx_dirac = checkpoint.get("zoom_kx_dirac")
    zoom_ky_dirac = checkpoint.get("zoom_ky_dirac")
    time_taken = checkpoint.get("time_taken")
    print(f"Minimum gap is: {zoom_min_gap} eV")
    print(f"Dirac point is at: ({zoom_kx_dirac}, {zoom_ky_dirac})")
    print(f"Time taken: {time_taken} seconds")