#!/usr/bin/env python3
"""
Preprocess the Recyclable and Household Waste Classification dataset.

What this script does:
1. Scans the dataset directory
2. Validates image files
3. Removes unreadable / corrupt / unsupported files from metadata
4. Optionally removes exact duplicates by MD5 hash
5. Creates clean metadata
6. Builds stratified train / val / test splits using class + source
7. Computes train-set mean / std
8. Saves everything to CSV / JSON for reproducible experiments

Expected dataset structure:
images/
  aerosol_cans/
    default/
    real_world/
  aluminum_food_cans/
    default/
    real_world/
  ...

Example usage:
python preprocess_waste_dataset.py \
    --data-root /path/to/images \
    --output-dir /path/to/output

Optional:
python preprocess_waste_dataset.py \
    --data-root /path/to/images \
    --output-dir /path/to/output \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --image-size-for-stats 128 \
    --batch-size-stats 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXPECTED_SOURCES = {"default", "real_world"}


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs(output_dir: Path) -> None:
    """Create output directories."""
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)


def file_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash for a file."""
    md5 = hashlib.md5()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def validate_image(file_path: Path) -> Dict[str, Optional[object]]:
    """
    Validate an image by attempting to open and verify it.

    Returns:
        A dict with:
        - valid
        - width
        - height
        - mode
        - error
    """
    try:
        with Image.open(file_path) as img:
            img.verify()

        with Image.open(file_path) as img:
            width, height = img.size
            mode = img.mode

        return {
            "valid": True,
            "width": width,
            "height": height,
            "mode": mode,
            "error": None,
        }
    except Exception as e:
        return {
            "valid": False,
            "width": None,
            "height": None,
            "mode": None,
            "error": str(e),
        }


def scan_dataset(data_root: Path) -> pd.DataFrame:
    """
    Scan dataset folders and collect metadata for all files.
    """
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    class_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {data_root}")

    rows: List[Dict[str, object]] = []

    for class_dir in tqdm(class_dirs, desc="Scanning classes"):
        class_name = class_dir.name
        source_dirs = [p for p in class_dir.iterdir() if p.is_dir()]
        source_names = {p.name for p in source_dirs}

        if source_names != EXPECTED_SOURCES:
            print(
                f"Warning: class '{class_name}' has source folders {source_names}, "
                f"expected {EXPECTED_SOURCES}"
            )

        for source_dir in source_dirs:
            source_name = source_dir.name

            for file_path in source_dir.iterdir():
                if not file_path.is_file():
                    continue

                ext = file_path.suffix.lower()
                size_bytes = file_path.stat().st_size

                row: Dict[str, object] = {
                    "filepath": str(file_path.resolve()),
                    "class_name": class_name,
                    "source": source_name,
                    "filename": file_path.name,
                    "extension": ext,
                    "size_bytes": size_bytes,
                }

                if ext not in VALID_EXTENSIONS:
                    row.update(
                        {
                            "valid": False,
                            "width": None,
                            "height": None,
                            "mode": None,
                            "md5": None,
                            "error": f"Unsupported extension: {ext}",
                        }
                    )
                    rows.append(row)
                    continue

                val_info = validate_image(file_path)
                row.update(val_info)

                if bool(val_info["valid"]):
                    try:
                        row["md5"] = file_md5(file_path)
                    except Exception as e:
                        row["valid"] = False
                        row["md5"] = None
                        row["error"] = f"MD5 error: {e}"
                else:
                    row["md5"] = None

                rows.append(row)

    return pd.DataFrame(rows)


def clean_metadata(
    df_all: pd.DataFrame,
    remove_exact_duplicates: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean metadata by:
    - keeping valid files only
    - removing zero-byte files
    - optionally removing exact duplicates by MD5
    """
    df_valid = df_all[df_all["valid"] == True].copy()  # noqa: E712
    df_valid = df_valid[df_valid["size_bytes"] > 0].copy()

    duplicates_df = pd.DataFrame(columns=df_valid.columns.tolist() + ["is_duplicate"])

    if remove_exact_duplicates:
        df_valid = df_valid.sort_values(["class_name", "source", "filepath"]).copy()
        df_valid["is_duplicate"] = df_valid.duplicated(subset=["md5"], keep="first")
        duplicates_df = df_valid[df_valid["is_duplicate"]].copy()
        df_clean = df_valid[~df_valid["is_duplicate"]].copy()
    else:
        df_clean = df_valid.copy()
        df_clean["is_duplicate"] = False

    invalid_df = df_all[df_all["valid"] != True].copy()  # noqa: E712
    return df_clean, invalid_df, duplicates_df


def add_class_indices(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """Add numeric class indices."""
    class_names = sorted(df_clean["class_name"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    df_clean = df_clean.copy()
    df_clean["class_idx"] = df_clean["class_name"].map(class_to_idx)
    return df_clean, class_to_idx, idx_to_class


def save_metadata_outputs(
    output_dir: Path,
    df_all: pd.DataFrame,
    df_clean: pd.DataFrame,
    invalid_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
) -> None:
    """Save scan logs and clean metadata."""
    df_all.to_csv(output_dir / "logs" / "raw_scan_metadata.csv", index=False)
    invalid_df.to_csv(output_dir / "logs" / "invalid_or_removed_files.csv", index=False)
    duplicates_df.to_csv(output_dir / "logs" / "removed_exact_duplicates.csv", index=False)
    df_clean.to_csv(output_dir / "clean_metadata.csv", index=False)

    with (output_dir / "class_to_idx.json").open("w") as f:
        json.dump(class_to_idx, f, indent=2)


def print_dataset_report(df_clean: pd.DataFrame) -> None:
    """Print a quick dataset report."""
    print("\n=== Clean Dataset Report ===")
    print(f"Total clean images: {len(df_clean)}")
    print(f"Number of classes: {df_clean['class_name'].nunique()}")

    print("\nImages per class:")
    print(df_clean["class_name"].value_counts().sort_index())

    print("\nImages per source:")
    print(df_clean["source"].value_counts().sort_index())

    print("\nImages per class and source:")
    class_source_counts = (
        df_clean.groupby(["class_name", "source"]).size().unstack(fill_value=0).sort_index()
    )
    print(class_source_counts)

    print("\nImage size summary:")
    print(df_clean[["width", "height"]].describe())

    print("\nImage mode counts:")
    print(df_clean["mode"].value_counts())


def create_splits(
    df_clean: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits preserving class + source balance.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    df = df_clean.copy()
    df["stratify_label"] = df["class_name"] + "__" + df["source"]

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=df["stratify_label"],
    )

    val_relative = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative),
        random_state=seed,
        stratify=temp_df["stratify_label"],
    )

    return train_df, val_df, test_df


def save_splits(output_dir: Path, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save train/val/test CSVs."""
    train_df.to_csv(output_dir / "splits" / "train.csv", index=False)
    val_df.to_csv(output_dir / "splits" / "val.csv", index=False)
    test_df.to_csv(output_dir / "splits" / "test.csv", index=False)


def print_split_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print split summary."""
    print("\n=== Split Report ===")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{split_name} class counts:")
        print(split_df["class_name"].value_counts().sort_index())

        print(f"\n{split_name} source counts:")
        print(split_df["source"].value_counts().sort_index())


class StatsDataset(Dataset):
    """Dataset used only for mean/std computation."""

    def __init__(self, df: pd.DataFrame, image_size: int = 128):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.df.loc[idx, "filepath"]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)
        return img


def compute_train_mean_std(
    train_df: pd.DataFrame,
    image_size_for_stats: int,
    batch_size_stats: int,
    num_workers: int,
) -> Dict[str, object]:
    """Compute train-set RGB mean and std."""
    dataset = StatsDataset(train_df, image_size=image_size_for_stats)
    loader = DataLoader(
        dataset,
        batch_size=batch_size_stats,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_pixels = 0

    for batch in tqdm(loader, desc="Computing mean/std"):
        b, c, h, w = batch.shape
        num_pixels += b * h * w
        channel_sum += batch.sum(dim=[0, 2, 3])
        channel_sq_sum += (batch ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sq_sum / num_pixels - mean**2)

    return {
        "image_size_used_for_stats": image_size_for_stats,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def save_train_mean_std(output_dir: Path, stats: Dict[str, object]) -> None:
    """Save train mean/std JSON."""
    with (output_dir / "train_mean_std.json").open("w") as f:
        json.dump(stats, f, indent=2)


def load_train_mean_std(output_dir: Path) -> Dict[str, object]:
    """Load train mean/std JSON."""
    with (output_dir / "train_mean_std.json").open("r") as f:
        return json.load(f)


def build_transforms(
    train_image_size: int,
    mean: List[float],
    std: List[float],
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build default train/eval transforms.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((train_image_size, train_image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((train_image_size, train_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, eval_transform


class WasteDataset(Dataset):
    """Dataset for training / validation / testing."""

    def __init__(self, csv_file: Path | str, transform: Optional[transforms.Compose] = None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.loc[idx]
        path = row["filepath"]
        label = int(row["class_idx"])

        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        return img, label


def build_dataloaders(
    output_dir: Path,
    train_image_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test dataloaders using saved splits and mean/std.
    """
    stats = load_train_mean_std(output_dir)
    train_mean = stats["mean"]
    train_std = stats["std"]

    train_transform, eval_transform = build_transforms(
        train_image_size=train_image_size,
        mean=train_mean,
        std=train_std,
    )

    train_csv = output_dir / "splits" / "train.csv"
    val_csv = output_dir / "splits" / "val.csv"
    test_csv = output_dir / "splits" / "test.csv"

    train_dataset = WasteDataset(train_csv, transform=train_transform)
    val_dataset = WasteDataset(val_csv, transform=eval_transform)
    test_dataset = WasteDataset(test_csv, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def sanity_check_dataloaders(output_dir: Path, train_image_size: int, batch_size: int, num_workers: int) -> None:
    """
    Optional sanity check to verify dataloaders work.
    """
    train_loader, val_loader, test_loader = build_dataloaders(
        output_dir=output_dir,
        train_image_size=train_image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    images, labels = next(iter(train_loader))
    print("\n=== Dataloader Sanity Check ===")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    print(f"Batch image shape: {tuple(images.shape)}")
    print(f"Batch label shape: {tuple(labels.shape)}")
    print(f"First 10 labels: {labels[:10].tolist()}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess waste classification dataset.")

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to the dataset images directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save metadata, splits, and stats.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--image-size-for-stats",
        type=int,
        default=128,
        help="Image size used when computing mean/std.",
    )
    parser.add_argument(
        "--batch-size-stats",
        type=int,
        default=64,
        help="Batch size used when computing mean/std.",
    )
    parser.add_argument(
        "--batch-size-loader",
        type=int,
        default=32,
        help="Batch size used for optional dataloader sanity check.",
    )
    parser.add_argument(
        "--train-image-size",
        type=int,
        default=128,
        help="Image size used for default train/eval transforms.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--no-remove-exact-duplicates",
        action="store_true",
        help="Disable removal of exact duplicate files by MD5.",
    )
    parser.add_argument(
        "--skip-dataloader-check",
        action="store_true",
        help="Skip final dataloader sanity check.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    set_seed(args.seed)
    ensure_dirs(output_dir)

    print("=== Configuration ===")
    print(f"DATA_ROOT: {data_root}")
    print(f"OUTPUT_DIR: {output_dir}")
    print(f"SEED: {args.seed}")

    print("\n[1/6] Scanning dataset...")
    df_all = scan_dataset(data_root)
    print(f"Total files scanned: {len(df_all)}")
    print("\nValidity summary:")
    print(df_all["valid"].value_counts(dropna=False))

    print("\n[2/6] Cleaning metadata...")
    df_clean, invalid_df, duplicates_df = clean_metadata(
        df_all,
        remove_exact_duplicates=not args.no_remove_exact_duplicates,
    )

    print(f"Valid images kept: {len(df_clean)}")
    print(f"Removed invalid/corrupt/unsupported: {len(invalid_df)}")
    print(f"Removed exact duplicates: {len(duplicates_df)}")

    print("\n[3/6] Adding class indices and saving metadata...")
    df_clean, class_to_idx, _ = add_class_indices(df_clean)
    save_metadata_outputs(output_dir, df_all, df_clean, invalid_df, duplicates_df, class_to_idx)
    print(f"Saved clean metadata to: {output_dir / 'clean_metadata.csv'}")
    print(f"Saved class mapping to: {output_dir / 'class_to_idx.json'}")

    print_dataset_report(df_clean)

    print("\n[4/6] Creating stratified train/val/test splits...")
    train_df, val_df, test_df = create_splits(
        df_clean=df_clean,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    save_splits(output_dir, train_df, val_df, test_df)
    print_split_report(train_df, val_df, test_df)

    print("\n[5/6] Computing train-set mean/std...")
    stats = compute_train_mean_std(
        train_df=train_df,
        image_size_for_stats=args.image_size_for_stats,
        batch_size_stats=args.batch_size_stats,
        num_workers=args.num_workers,
    )
    save_train_mean_std(output_dir, stats)
    print(f"Train mean: {stats['mean']}")
    print(f"Train std:  {stats['std']}")
    print(f"Saved mean/std to: {output_dir / 'train_mean_std.json'}")

    print("\n[6/6] Finalizing...")
    if not args.skip_dataloader_check:
        sanity_check_dataloaders(
            output_dir=output_dir,
            train_image_size=args.train_image_size,
            batch_size=args.batch_size_loader,
            num_workers=args.num_workers,
        )

    print("\nDone.")
    print("\nSaved files:")
    print(f"- {output_dir / 'clean_metadata.csv'}")
    print(f"- {output_dir / 'class_to_idx.json'}")
    print(f"- {output_dir / 'train_mean_std.json'}")
    print(f"- {output_dir / 'splits' / 'train.csv'}")
    print(f"- {output_dir / 'splits' / 'val.csv'}")
    print(f"- {output_dir / 'splits' / 'test.csv'}")
    print(f"- {output_dir / 'logs' / 'raw_scan_metadata.csv'}")
    print(f"- {output_dir / 'logs' / 'invalid_or_removed_files.csv'}")
    print(f"- {output_dir / 'logs' / 'removed_exact_duplicates.csv'}")


if __name__ == "__main__":
    main()