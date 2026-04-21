import argparse
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print upper/lower Dirac points and gaps from checkpoints across strains."
    )
    parser.add_argument(
        "--strains",
        type=float,
        nargs="+",
        required=True,
        help="Space-separated strain values (example: --strains 0.0 5.0 10.0).",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoint*.pkl files (default: checkpoints).",
    )
    return parser.parse_args()


def format_point(kx, ky):
    if kx is None or ky is None:
        return "N/A"
    return f"({kx:.8f}, {ky:.8f})"


def format_gap(gap):
    if gap is None:
        return "N/A"
    return f"{gap:.8f}"


def read_checkpoint(checkpoints_dir, strain):
    strain_str = f"{strain:.1f}"
    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint{strain_str}.pkl")
    if not os.path.exists(checkpoint_path):
        return strain_str, None
    with open(checkpoint_path, "rb") as f:
        return strain_str, pickle.load(f)


def main():
    args = parse_args()

    print("Upper/Lower Dirac summary from checkpoints")
    print("=" * 230)
    print(
        f"{'strain(%)':>10}  "
        f"{'coarse upper (kx, ky)':>30}  {'coarse upper gap':>16}  "
        f"{'coarse lower (kx, ky)':>30}  {'coarse lower gap':>16}  "
        f"{'fine upper (kx, ky)':>30}  {'fine upper gap':>14}  "
        f"{'fine lower (kx, ky)':>30}  {'fine lower gap':>14}  "
        f"status"
    )
    print("-" * 230)

    for strain in args.strains:
        strain_str, checkpoint = read_checkpoint(args.checkpoints_dir, strain)
        if checkpoint is None:
            print(
                f"{strain_str:>10}  {'N/A':>30}  {'N/A':>16}  {'N/A':>30}  {'N/A':>16}  "
                f"{'N/A':>30}  {'N/A':>14}  {'N/A':>30}  {'N/A':>14}  missing checkpoint"
            )
            continue

        coarse_upper_point = format_point(
            checkpoint.get("upper_kx_dirac_coarse"), checkpoint.get("upper_ky_dirac_coarse")
        )
        coarse_upper_gap = format_gap(checkpoint.get("upper_min_gap_coarse"))
        coarse_lower_point = format_point(
            checkpoint.get("lower_kx_dirac_coarse"), checkpoint.get("lower_ky_dirac_coarse")
        )
        coarse_lower_gap = format_gap(checkpoint.get("lower_min_gap_coarse"))

        fine_upper_point = format_point(
            checkpoint.get("upper_kx_dirac"), checkpoint.get("upper_ky_dirac")
        )
        fine_upper_gap = format_gap(checkpoint.get("upper_min_gap"))
        fine_lower_point = format_point(
            checkpoint.get("lower_kx_dirac"), checkpoint.get("lower_ky_dirac")
        )
        fine_lower_gap = format_gap(checkpoint.get("lower_min_gap"))

        has_any = any(
            value != "N/A"
            for value in [
                coarse_upper_point,
                coarse_upper_gap,
                coarse_lower_point,
                coarse_lower_gap,
                fine_upper_point,
                fine_upper_gap,
                fine_lower_point,
                fine_lower_gap,
            ]
        )
        status = "ok" if has_any else "no dirac keys"

        print(
            f"{strain_str:>10}  {coarse_upper_point:>30}  {coarse_upper_gap:>16}  "
            f"{coarse_lower_point:>30}  {coarse_lower_gap:>16}  "
            f"{fine_upper_point:>30}  {fine_upper_gap:>14}  "
            f"{fine_lower_point:>30}  {fine_lower_gap:>14}  {status}"
        )

    print("=" * 230)


if __name__ == "__main__":
    main()
