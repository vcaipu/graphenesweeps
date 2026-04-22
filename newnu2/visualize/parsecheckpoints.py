import argparse
import csv
import json
import math
import os
import pickle
import re


# Keep lattice constants consistent with masterscript.py
A0_BOHR = 4.625923576
POISSON_NU = 0.165
SQRT3 = math.sqrt(3.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export checkpoint data to CSV (one row per strain)."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoint*.pkl files (default: checkpoints).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoint_summary.csv",
        help="Output CSV path (default: checkpoint_summary.csv).",
    )
    parser.add_argument(
        "--strains",
        type=float,
        nargs="*",
        default=None,
        help="Optional explicit strain list. If omitted, all checkpoint*.pkl are exported.",
    )
    return parser.parse_args()


def discover_checkpoint_paths(checkpoints_dir, strains=None):
    if strains:
        paths = []
        for strain in strains:
            strain_str = f"{strain:.1f}"
            p = os.path.join(checkpoints_dir, f"checkpoint{strain_str}.pkl")
            if os.path.exists(p):
                paths.append(p)
            else:
                print(f"Warning: missing checkpoint for strain {strain_str}: {p}")
        return paths

    if not os.path.isdir(checkpoints_dir):
        return []
    paths = []
    pattern = re.compile(r"^checkpoint([+-]?\d+(?:\.\d+)?)\.pkl$")
    for name in sorted(os.listdir(checkpoints_dir)):
        if pattern.match(name):
            paths.append(os.path.join(checkpoints_dir, name))
    return paths


def parse_strain_from_path(path):
    name = os.path.basename(path)
    m = re.match(r"^checkpoint([+-]?\d+(?:\.\d+)?)\.pkl$", name)
    if not m:
        return None
    return float(m.group(1))


def reciprocal_vectors_tpiba(strain):
    """
    Reciprocal vectors in the same tpiba-style coordinate system used by the script.
    For zigzag strain in masterscript:
      a1 = a*(F11, 0), a2 = a*(F11/2, sqrt(3)/2*F22)
    with epsilon = strain/100 and F11, F22 below.

    Important unit note:
    QE k-points read from data-file-schema.xml are in "tpiba" units, i.e. scaled by
    (2*pi/alat). With CELL_PARAMETERS, alat follows |a1|, so tpiba normalization
    introduces an extra factor F11 in reciprocal components.

    In these tpiba coordinates:
      b1 = ( 1, -F11/(sqrt(3)*F22) )
      b2 = ( 0,  2*F11/(sqrt(3)*F22) )
    """
    eps = strain / 100.0
    f11 = 1.0 + eps
    f22 = 1.0 - (POISSON_NU * eps)
    ratio = f11 / f22
    b1 = (1.0, -ratio / SQRT3)
    b2 = (0.0, 2.0 * ratio / SQRT3)
    return b1, b2


def min_periodic_distance(p1, p2, b1, b2, neighbor_window=3):
    """
    Return shortest periodic distance between p1 and p2 under lattice shifts m*b1+n*b2.

    Uses a closest-vector approach:
      1) Solve B * [m, n] ~= (p1 - p2), where B=[b1 b2]
      2) Search small neighborhood around rounded fractional shift
    This is robust near Brillouin-zone boundaries and avoids missing the
    nearest image due to a too-small fixed global brute-force window.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    # Invert 2x2 matrix B = [[b1x, b2x], [b1y, b2y]]
    det = b1[0] * b2[1] - b1[1] * b2[0]
    if abs(det) < 1e-14:
        raise ValueError("Reciprocal basis is singular; cannot compute periodic distance.")

    inv11 = b2[1] / det
    inv12 = -b2[0] / det
    inv21 = -b1[1] / det
    inv22 = b1[0] / det

    m_float = inv11 * dx + inv12 * dy
    n_float = inv21 * dx + inv22 * dy

    m0 = int(round(m_float))
    n0 = int(round(n_float))

    best_d = None
    best = None
    for dm in range(-neighbor_window, neighbor_window + 1):
        for dn in range(-neighbor_window, neighbor_window + 1):
            m = m0 + dm
            n = n0 + dn
            sx = p2[0] + m * b1[0] + n * b2[0]
            sy = p2[1] + m * b1[1] + n * b2[1]
            ddx = p1[0] - sx
            ddy = p1[1] - sy
            d = math.hypot(ddx, ddy)
            if best_d is None or d < best_d:
                best_d = d
                best = (m, n, sx, sy)

    return best_d, best


def maybe_compute_dirac_distance(row, checkpoint, strain):
    b1, b2 = reciprocal_vectors_tpiba(strain)

    def add_distance(prefix, k1x, k1y, k2x, k2y):
        if None in (k1x, k1y, k2x, k2y):
            row[f"{prefix}_dirac_periodic_distance"] = ""
            row[f"{prefix}_dirac_shift_m"] = ""
            row[f"{prefix}_dirac_shift_n"] = ""
            return
        d, best = min_periodic_distance((k1x, k1y), (k2x, k2y), b1, b2)
        row[f"{prefix}_dirac_periodic_distance"] = d
        row[f"{prefix}_dirac_shift_m"] = best[0]
        row[f"{prefix}_dirac_shift_n"] = best[1]

    add_distance(
        "coarse",
        checkpoint.get("upper_kx_dirac_coarse"),
        checkpoint.get("upper_ky_dirac_coarse"),
        checkpoint.get("lower_kx_dirac_coarse"),
        checkpoint.get("lower_ky_dirac_coarse"),
    )
    add_distance(
        "fine",
        checkpoint.get("upper_kx_dirac"),
        checkpoint.get("upper_ky_dirac"),
        checkpoint.get("lower_kx_dirac"),
        checkpoint.get("lower_ky_dirac"),
    )


def serialize_value(v):
    if isinstance(v, (tuple, list, dict)):
        return json.dumps(v)
    return v


def main():
    args = parse_args()
    paths = discover_checkpoint_paths(args.checkpoints_dir, args.strains)
    if not paths:
        print(f"Error: no checkpoint files found in '{args.checkpoints_dir}'.")
        return

    rows = []
    all_fields = {"strain"}
    derived_fields = {
        "coarse_dirac_periodic_distance",
        "coarse_dirac_shift_m",
        "coarse_dirac_shift_n",
        "fine_dirac_periodic_distance",
        "fine_dirac_shift_m",
        "fine_dirac_shift_n",
    }
    all_fields.update(derived_fields)

    for path in paths:
        strain = parse_strain_from_path(path)
        if strain is None:
            print(f"Warning: skipped unrecognized checkpoint filename: {path}")
            continue
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        if not isinstance(checkpoint, dict):
            print(f"Warning: skipped non-dict checkpoint: {path}")
            continue

        row = {"strain": f"{strain:.1f}"}
        for k, v in checkpoint.items():
            row[k] = serialize_value(v)
            all_fields.add(k)

        maybe_compute_dirac_distance(row, checkpoint, strain)
        rows.append(row)

    if not rows:
        print("Error: no valid checkpoint rows to export.")
        return

    # Keep key columns first, then remaining checkpoint keys sorted.
    priority = [
        "strain",
        "upper_kx_dirac_coarse",
        "upper_ky_dirac_coarse",
        "lower_kx_dirac_coarse",
        "lower_ky_dirac_coarse",
        "coarse_dirac_periodic_distance",
        "coarse_dirac_shift_m",
        "coarse_dirac_shift_n",
        "upper_kx_dirac",
        "upper_ky_dirac",
        "lower_kx_dirac",
        "lower_ky_dirac",
        "fine_dirac_periodic_distance",
        "fine_dirac_shift_m",
        "fine_dirac_shift_n",
    ]
    remaining = sorted(f for f in all_fields if f not in priority)
    fieldnames = priority + remaining

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: float(r["strain"])):
            writer.writerow(row)

    print(f"Wrote CSV with {len(rows)} rows: {args.output}")


if __name__ == "__main__":
    main()