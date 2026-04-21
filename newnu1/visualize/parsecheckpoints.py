import argparse
import os
import pickle
from pprint import pformat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print checkpoint contents in a human-readable format."
    )
    parser.add_argument(
        "--strain",
        type=float,
        required=True,
        help="Strain value used in the run (example: 10.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    strain_str = f"{args.strain:.1f}"
    prefix = f"graphene{strain_str}"
    zoom_prefix = f"{prefix}zoom"
    checkpoint_path = os.path.join("checkpoints", f"checkpoint{strain_str}.pkl")

    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint file not found at '{checkpoint_path}'")
        return

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    print("Checkpoint summary")
    print("=" * 72)
    print(f"Strain:      {strain_str}%")
    print(f"Prefix:      {prefix}")
    print(f"Zoom prefix: {zoom_prefix}")
    print(f"File:        {checkpoint_path}")
    print("=" * 72)

    if not checkpoint:
        print("(Checkpoint is empty)")
        return

    for key in sorted(checkpoint.keys()):
        value = checkpoint[key]
        pretty_value = pformat(value, width=100, compact=False)
        print(f"{key}:")
        for line in pretty_value.splitlines():
            print(f"  {line}")
        print()


if __name__ == "__main__":
    main()