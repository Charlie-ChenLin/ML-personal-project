from __future__ import annotations

import argparse
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pp_forecast.q1_dataset import build_q1_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Q1 monthly dataset (PP price prediction).")
    p.add_argument("--data-dir", type=Path, default=Path("PP数据"))
    p.add_argument("--target-metric", choices=["mean", "last"], default="mean")
    p.add_argument("--include-futures", action="store_true", help="Include file 13 futures factor.")
    p.add_argument(
        "--engineer-features",
        action="store_true",
        help="Add extra engineered features (rolling stats, momentum, spreads, ratios).",
    )
    p.add_argument(
        "--strong-threshold",
        type=float,
        default=0.05,
        help="Return threshold for big up/down (e.g. 0.05 means 5%).",
    )
    p.add_argument(
        "--flat-threshold",
        type=float,
        default=0.005,
        help="Return threshold treated as flat/no-change (e.g. 0.005 means ±0.5%).",
    )
    p.add_argument("--output", type=Path, default=Path("outputs/datasets/q1_dataset.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = build_q1_dataset(
        data_dir=args.data_dir,
        target_metric=args.target_metric,
        include_futures=args.include_futures,
        engineer_features=args.engineer_features,
        strong_threshold=args.strong_threshold,
        flat_threshold=args.flat_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds.data.to_csv(args.output, index=False)
    print(f"[OK] wrote dataset: {args.output}")
    print(f"  rows={len(ds.data)}, cols={len(ds.data.columns)}")
    print(
        f"  target_metric={ds.target_metric}, include_futures={ds.include_futures}, "
        f"engineer_features={ds.engineer_features}, "
        f"strong_threshold={ds.strong_threshold}, flat_threshold={ds.flat_threshold}"
    )


if __name__ == "__main__":
    main()
