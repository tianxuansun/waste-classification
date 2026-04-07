#!/usr/bin/env python3
"""
Split a waste classification dataset into training and testing sets.
Within each category, images under ``default`` and ``real_world`` are split by ratio so both
splits contain both domains with no overlap between train and test.
Output layout: ``data/training data/`` and ``data/testing data/``, mirroring the class folder tree.
If the archive has a double ``images/images/...`` nest, the script picks the inner class root automatically.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}


def pick_existing_subdir(parent: Path, *candidates: str) -> Path | None:
    """Return the first ``parent/name`` that exists and is a directory."""
    for name in candidates:
        p = parent / name
        if p.is_dir():
            return p
    return None


def _folder_has_default_or_real_world(class_dir: Path) -> bool:
    d = pick_existing_subdir(class_dir, "default", "Default")
    rw = pick_existing_subdir(class_dir, "real_world", "real-world", "real world", "Real_world")
    return d is not None or rw is not None


def resolve_class_root(source: Path) -> Path:
    """
    Some archives nest twice: ``.../images/images/<class>/default|real_world``.
    If ``source/images`` exists and looks like the real class root, use that path.
    """
    nested = source / "images"
    if not nested.is_dir():
        return source
    for child in nested.iterdir():
        if child.is_dir() and _folder_has_default_or_real_world(child):
            return nested
    return source


def list_images(folder: Path, *, recursive: bool) -> list[Path]:
    """Collect image paths under ``folder``. Use recursive=True if images live in subfolders."""
    if not folder.is_dir():
        return []
    candidates: list[Path]
    if recursive:
        candidates = [p for p in folder.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in folder.iterdir() if p.is_file()]
    out = [p for p in candidates if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(out, key=lambda p: str(p).lower())


def split_indices(n: int, train_ratio: float, rng: random.Random) -> tuple[set[int], set[int]]:
    """Return disjoint train/test index sets whose union is ``range(n)``."""
    if n == 0:
        return set(), set()
    indices = list(range(n))
    rng.shuffle(indices)
    k_train = int(round(n * train_ratio))
    k_train = max(0, min(n, k_train))
    train_set = set(indices[:k_train])
    test_set = set(indices[k_train:])
    return train_set, test_set


def copy_files(paths: list[Path], indices: set[int], dest_subdir: Path) -> None:
    if not indices:
        return
    dest_subdir.mkdir(parents=True, exist_ok=True)
    for i in sorted(indices):
        src = paths[i]
        dst = dest_subdir / src.name
        if dst.exists():
            stem, suf = src.stem, src.suffix
            n = 1
            while dst.exists():
                dst = dest_subdir / f"{stem}_{n}{suf}"
                n += 1
        shutil.copy2(src, dst)


def process_category(
    category_dir: Path,
    train_root: Path,
    test_root: Path,
    train_ratio: float,
    rng: random.Random,
    *,
    recursive: bool,
    verbose: bool,
) -> tuple[int, int]:
    name = category_dir.name
    default_dir = pick_existing_subdir(category_dir, "default", "Default")
    rw_dir = pick_existing_subdir(category_dir, "real_world", "real world")

    default_files = list_images(default_dir, recursive=recursive) if default_dir else []
    rw_files = list_images(rw_dir, recursive=recursive) if rw_dir else []

    if verbose:
        print(f"  [{name}] default: {len(default_files)} images, real_world: {len(rw_files)} images")

    d_train_idx, d_test_idx = split_indices(len(default_files), train_ratio, rng)
    r_train_idx, r_test_idx = split_indices(len(rw_files), train_ratio, rng)

    copy_files(default_files, d_train_idx, train_root / name / "default")
    copy_files(default_files, d_test_idx, test_root / name / "default")
    copy_files(rw_files, r_train_idx, train_root / name / "real_world")
    copy_files(rw_files, r_test_idx, test_root / name / "real_world")

    return len(default_files), len(rw_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split default/real_world images into train and test (no overlap); writes under data/."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("images"),
        help="Root folder containing one subdirectory per class/category.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help='Output root; creates subfolders "training data" and "testing data".',
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of images in the training set (e.g. 0.8 for ~80%% train, 20%% test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only look for images directly inside default/ and real_world/ (not in subfolders).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print image counts per category.",
    )
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        parser.error("--train-ratio must be strictly between 0 and 1 (e.g. 0.8).")

    source: Path = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source path is missing or not a directory: {source}")

    resolved = resolve_class_root(source)
    if resolved != source:
        print(f"Using class root inside nested folder: {resolved}")

    source = resolved

    out = args.output.resolve()
    train_root = out / "training data"
    test_root = out / "testing data"

    if train_root.exists() or test_root.exists():
        raise SystemExit(
            f"Output already exists; remove it or use a different --output: {train_root} / {test_root}"
        )

    rng = random.Random(args.seed)

    categories = [p for p in source.iterdir() if p.is_dir()]
    if not categories:
        raise SystemExit(f"No category subdirectories found under {source}")

    recursive = not args.no_recursive
    total_default = 0
    total_rw = 0
    for cat in sorted(categories, key=lambda p: p.name):
        nd, nr = process_category(
            cat,
            train_root,
            test_root,
            args.train_ratio,
            rng,
            recursive=recursive,
            verbose=args.verbose,
        )
        total_default += nd
        total_rw += nr

    if total_default == 0 and total_rw == 0:
        raise SystemExit(
            f"No image files found under any class folder's 'default' or 'real_world' subfolders.\n"
            f"  Source: {source}\n"
            f"  Expected layout: {source}/<class_name>/default/... and .../real_world/...\n"
            f"  Pass the directory that directly contains the class folders (often named 'images').\n"
            f"  If images are only in subfolders of default/real_world, run without --no-recursive (default)."
        )

    print(f"Done. Processed {len(categories)} categories ({total_default} default + {total_rw} real_world images).")
    print(f"Training: {train_root}")
    print(f"Testing:  {test_root}")


if __name__ == "__main__":
    main()
