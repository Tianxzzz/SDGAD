"""
Create the uci_mix dataset by injecting typed anomalies into the original uci data.

"""

import argparse

from flexible_anomaly_generator import FlexibleAnomalyGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate the `uci_mix` anomaly dataset.")
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="random seed used for the time split and sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed_data",
        help="destination folder for the generated dataset",
    )
    args = parser.parse_args()

    generator = FlexibleAnomalyGenerator(
        dataset_name="lastfm",
        val_ratio=0.2,
        test_ratio=0.4,
        seed=args.seed,
        enable_contextual=False,
    )
    generator.load_and_split_data()

    anomaly_config = {
        "train": [
            {"type": "temporal", "ratio": 0.001, "label": 1},
        ],
        "val": [
            {"type": "temporal", "ratio": 0.001, "label": 1},
        ],
        "test": [
            {"type": "temporal", "ratio": 0.0005, "label": 1},
            {"type": "structural", "ratio": 0.0005, "label": 2},
            # {"type": "mixed", "ratio": 0.00033, "label": 3, "temporal_ratio": 0.5},
        ],
    }

    final_dataset = generator.apply_anomaly_config(anomaly_config)
    generator.save_dataset(final_dataset, "lastfm_mix_0.001", output_dir=args.output_dir)

    print("\n✅ `uci_mix` generated.")
    print(f"  - Saved under: {args.output_dir}/uci_mix_0.001")
    print("  - Temporal anomalies (T) keep label 1, structural (S) use 2, mixed (T-S) use 3.")
    print("  - Use `--dataset_name uci_mix` when you want to train or evaluate on it.")


if __name__ == "__main__":
    main()
