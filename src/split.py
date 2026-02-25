"""
Patient-level train/validation/test split for MEETI / MIMIC-IV-ECG.

Usage:
    python src/split.py --data_dir data/MEETI \
                        --output_dir splits/ \
                        [--train_ratio 0.70] \
                        [--valid_ratio 0.10] \
                        [--test_ratio 0.20] \
                        [--stratify_col <column_name>] \
                        [--seed 42]

    # Or, if you already have a record_list.csv:
    python src/split.py --record_list <path/to/record_list.csv> \
                        --output_dir splits/ \
                        [--seed 42]

The script splits at the *patient* level (subject_id) so that no patient
appears in more than one split, preventing data leakage.
"""

import argparse
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate patient-level train/valid/test splits for MEETI."
    )
    parser.add_argument(
        "--record_list",
        type=str,
        default=None,
        help="Path to record_list.csv (must contain a 'subject_id' column).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to the MEETI data directory (e.g. data/MEETI). "
            "The directory is scanned to build a record list automatically. "
            "Exactly one of --record_list or --data_dir must be provided."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="splits",
        help="Directory where train.csv, valid.csv, test.csv will be written.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Fraction of patients assigned to the training set (default: 0.70).",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.10,
        help="Fraction of patients assigned to the validation set (default: 0.10).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.20,
        help="Fraction of patients assigned to the test set (default: 0.20).",
    )
    parser.add_argument(
        "--stratify_col",
        type=str,
        default=None,
        help=(
            "Optional column in record_list.csv to stratify on. "
            "When provided, the most frequent value per patient is used as the "
            "stratification label (e.g. a primary diagnosis code)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def validate_ratios(train: float, valid: float, test: float) -> None:
    total = round(train + valid + test, 6)
    if total != 1.0:
        raise ValueError(
            f"train_ratio + valid_ratio + test_ratio must equal 1.0, got {total}."
        )
    for name, val in [("train", train), ("valid", valid), ("test", test)]:
        if not (0.0 < val < 1.0):
            raise ValueError(f"{name}_ratio must be in (0, 1), got {val}.")


def get_patient_label(
    records: pd.DataFrame, subject_col: str, label_col: str
) -> pd.Series:
    """Return the most frequent label per patient for stratification."""
    return (
        records.groupby(subject_col)[label_col]
        .agg(lambda x: x.value_counts().index[0])
    )


def build_record_list(data_dir: str) -> pd.DataFrame:
    """Scan a MEETI / MIMIC-IV-ECG directory tree and return a record DataFrame.

    Expected layout:  <data_dir>/p<bucket>/p<subject_id>/s<study_id>/<study_id>.mat
    """
    rows: list[dict[str, str | int]] = []
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    for bucket_dir in sorted(data_path.iterdir()):
        if not bucket_dir.is_dir():
            continue
        for subject_dir in sorted(bucket_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            match = re.match(r"p(\d+)", subject_dir.name)
            if not match:
                continue
            subject_id = int(match.group(1))
            for study_dir in sorted(subject_dir.iterdir()):
                if not study_dir.is_dir():
                    continue
                study_match = re.match(r"s(\d+)", study_dir.name)
                if not study_match:
                    continue
                study_id = int(study_match.group(1))
                rows.append(
                    {
                        "subject_id": subject_id,
                        "study_id": study_id,
                        "path": str(study_dir.relative_to(data_path)),
                    }
                )

    if not rows:
        raise RuntimeError(
            f"No records found under {data_dir}. "
            "Expected layout: p<bucket>/p<subject_id>/s<study_id>/"
        )

    df = pd.DataFrame(rows)
    print(f"Built record list from '{data_dir}': {len(df):,} records found.")
    return df


def split_patients(
    patient_ids: np.ndarray,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    stratify_labels: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split patient IDs into train / valid / test arrays."""
    temp_size = valid_ratio + test_ratio
    test_fraction_of_temp = test_ratio / temp_size

    # First split: train vs (valid + test)
    if stratify_labels is not None:
        try:
            train_ids, temp_ids, train_strat, temp_strat = train_test_split(
                patient_ids,
                stratify_labels,
                test_size=temp_size,
                random_state=seed,
                stratify=stratify_labels,
            )
        except ValueError:
            warnings.warn(
                "Stratified first split failed (likely rare classes). "
                "Falling back to non-stratified split."
            )
            stratify_labels = None

    if stratify_labels is None:
        train_ids, temp_ids = train_test_split(
            patient_ids,
            test_size=temp_size,
            random_state=seed,
        )
        temp_strat = None
    # Second split: valid vs test
    if stratify_labels is not None:
        try:
            valid_ids, test_ids = train_test_split(
                temp_ids,
                test_size=test_fraction_of_temp,
                random_state=seed,
                stratify=temp_strat,
            )
        except ValueError:
            warnings.warn(
                "Stratified second split failed (likely rare classes). "
                "Falling back to non-stratified split for valid/test."
            )
            valid_ids, test_ids = train_test_split(
                temp_ids,
                test_size=test_fraction_of_temp,
                random_state=seed,
            )
    else:
        valid_ids, test_ids = train_test_split(
            temp_ids,
            test_size=test_fraction_of_temp,
            random_state=seed,
        )

    return train_ids, valid_ids, test_ids


def main() -> None:
    args = parse_args()

    if args.record_list is None and args.data_dir is None:
        raise ValueError("Exactly one of --record_list or --data_dir must be provided.")
    if args.record_list is not None and args.data_dir is not None:
        raise ValueError("Provide only one of --record_list or --data_dir, not both.")

    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)

    if args.record_list is not None:
        records = pd.read_csv(args.record_list)
    else:
        records = build_record_list(args.data_dir)

    if "subject_id" not in records.columns:
        raise ValueError(
            "'subject_id' column not found in record_list.csv. "
            f"Available columns: {list(records.columns)}"
        )

    patient_ids = records["subject_id"].unique()
    print(f"Total unique patients : {len(patient_ids):,}")
    print(f"Total records         : {len(records):,}")

    stratify_labels: np.ndarray | None = None
    if args.stratify_col is not None:
        if args.stratify_col not in records.columns:
            raise ValueError(
                f"stratify_col '{args.stratify_col}' not found in record_list.csv."
            )
        patient_label_series = get_patient_label(
            records, "subject_id", args.stratify_col
        )
        stratify_labels = patient_label_series.loc[patient_ids].values
        print(f"Stratifying on column : '{args.stratify_col}'")

    train_ids, valid_ids, test_ids = split_patients(
        patient_ids=patient_ids,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        stratify_labels=stratify_labels,
        seed=args.seed,
    )

    train_df = records[records["subject_id"].isin(train_ids)].copy()
    valid_df = records[records["subject_id"].isin(valid_ids)].copy()
    test_df = records[records["subject_id"].isin(test_ids)].copy()

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.csv")
    valid_path = os.path.join(args.output_dir, "valid.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSplit summary (patients / records):")
    print(f"  Train : {train_df['subject_id'].nunique():>6,} patients  |  {len(train_df):>8,} records  ({len(train_df)/len(records)*100:.1f}%)")
    print(f"  Valid : {valid_df['subject_id'].nunique():>6,} patients  |  {len(valid_df):>8,} records  ({len(valid_df)/len(records)*100:.1f}%)")
    print(f"  Test  : {test_df['subject_id'].nunique():>6,} patients  |  {len(test_df):>8,} records  ({len(test_df)/len(records)*100:.1f}%)")

    assert set(train_ids).isdisjoint(valid_ids), "Patient overlap: train ∩ valid"
    assert set(train_ids).isdisjoint(test_ids), "Patient overlap: train ∩ test"
    assert set(valid_ids).isdisjoint(test_ids), "Patient overlap: valid ∩ test"
    print("\nVerification passed: no patient overlap across splits.")
    print(f"\nSplit manifests written to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
