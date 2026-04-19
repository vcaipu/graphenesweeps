import argparse
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Dirac point locations from multiple checkpoint files."
    )
    parser.add_argument(
        "--strains",
        type=float,
        nargs="+",
        required=True,
        help="Space-separated strain values (example: --strains 0.0 10.0 15.0).",
    )
    return parser.parse_args()


def read_checkpoint(strain):
    strain_str = f"{strain:.1f}"
    checkpoint_path = os.path.join("checkpoints", f"checkpoint{strain_str}.pkl")

    if not os.path.exists(checkpoint_path):
        return strain_str, checkpoint_path, None

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    return strain_str, checkpoint_path, checkpoint


def format_point(kx, ky):
    if kx is None or ky is None:
        return "N/A"
    return f"({kx:.8f}, {ky:.8f})"

def format_gap(gap):
    if gap is None:
        return "N/A"
    return f"{gap:.8f}"


def main():
    args = parse_args()

    print("Dirac point summary")
    print("=" * 90)
    print(
        f"{'strain(%)':>10}  {'coarse_dirac (kx, ky)':>30}  {'zoom_dirac (kx, ky)':>30}  {'zoom_bandgap (eV)':>18}  status"
    )
    print("-" * 120)

    for strain in args.strains:
        strain_str, checkpoint_path, checkpoint = read_checkpoint(strain)

        if checkpoint is None:
            print(
                f"{strain_str:>10}  {'N/A':>30}  {'N/A':>30}  missing file: {checkpoint_path}"
            )
            continue

        coarse_kx = checkpoint.get("kx_dirac")
        coarse_ky = checkpoint.get("ky_dirac")
        zoom_kx = checkpoint.get("zoom_kx_dirac")
        zoom_ky = checkpoint.get("zoom_ky_dirac")
        zoom_gap = checkpoint.get("zoom_min_gap")

        coarse_point = format_point(coarse_kx, coarse_ky)
        zoom_point = format_point(zoom_kx, zoom_ky)
        zoom_gap_text = format_gap(zoom_gap)

        if coarse_point == "N/A" and zoom_point == "N/A":
            status = "checkpoint found, no dirac keys"
        else:
            status = "ok"

        print(
            f"{strain_str:>10}  {coarse_point:>30}  {zoom_point:>30}  {zoom_gap_text:>18}  {status}"
        )

    print("=" * 120)


if __name__ == "__main__":
    main()
