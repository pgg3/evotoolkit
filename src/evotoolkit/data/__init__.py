# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
EvoToolkit data management module.

Provides automatic dataset downloading from GitHub releases.
"""

from pathlib import Path
from typing import Optional

from .downloader import ensure_dataset_downloaded, DownloadError
from .constants import DATASET_CATEGORIES


__all__ = ["get_dataset_path", "DownloadError", "list_available_datasets"]


def get_dataset_path(category: str, data_dir: Optional[Path | str] = None) -> Path:
    """
    Get path to dataset category, downloading if necessary.

    This function returns the base directory for a dataset category (e.g., "scientific_regression").
    Individual datasets within the category can be accessed as subdirectories.

    The function will automatically download datasets from GitHub releases on first use.

    Args:
        category: Dataset category name (e.g., "scientific_regression")
        data_dir: Custom data directory. If None, defaults to ~/.evotool/data/

    Returns:
        Path to the category base directory containing all datasets

    Raises:
        ValueError: If category is unknown
        DownloadError: If download fails
        FileNotFoundError: If dataset not found after download

    Example:
        >>> from evotoolkit.data import get_dataset_path
        >>> base_dir = get_dataset_path('scientific_regression')
        >>> bactgrow_path = base_dir / 'bactgrow'
        >>> train_csv = bactgrow_path / 'train.csv'
    """
    if category not in DATASET_CATEGORIES:
        available = list(DATASET_CATEGORIES.keys())
        raise ValueError(
            f"Unknown dataset category: {category}. Available categories: {available}"
        )

    # Convert string to Path if needed
    if data_dir is not None and isinstance(data_dir, str):
        data_dir = Path(data_dir)

    # Determine base directory
    if data_dir is None:
        base_dir = Path.home() / ".evotool" / "data"
    else:
        base_dir = Path(data_dir)

    category_dir = base_dir / category

    # Check if category directory exists with any datasets
    if category_dir.exists() and any(category_dir.iterdir()):
        # Category already downloaded, return it
        return category_dir

    # Need to download - we'll download by requesting the first dataset
    # The downloader will fetch the entire category
    category_config = DATASET_CATEGORIES[category]
    first_dataset = list(category_config["datasets"].keys())[0]

    # This will trigger download of entire category if needed
    ensure_dataset_downloaded(category, first_dataset, base_dir)

    return category_dir


def list_available_datasets(category: str) -> dict:
    """
    List all available datasets in a category.

    Args:
        category: Dataset category name

    Returns:
        Dictionary mapping dataset names to their metadata

    Raises:
        ValueError: If category is unknown

    Example:
        >>> from evotoolkit.data import list_available_datasets
        >>> datasets = list_available_datasets('scientific_regression')
        >>> print(datasets.keys())
        dict_keys(['bactgrow', 'oscillator1', 'oscillator2', 'stressstrain'])
    """
    if category not in DATASET_CATEGORIES:
        available = list(DATASET_CATEGORIES.keys())
        raise ValueError(
            f"Unknown dataset category: {category}. Available categories: {available}"
        )

    return DATASET_CATEGORIES[category]["datasets"].copy()
