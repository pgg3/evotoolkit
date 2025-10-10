# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Dataset metadata and download configuration.
"""

from typing import Dict, List

# GitHub release configuration
GITHUB_REPO = "pgg3/evotoolkit"
DATA_VERSION = "v1.0.0"

# Scientific regression datasets metadata
SCIENTIFIC_REGRESSION_DATASETS = {
    "bactgrow": {
        "description": "E. Coli bacterial growth rate prediction",
        "required_files": ["train.csv", "test_id.csv", "test_ood.csv"],
    },
    "oscillator1": {
        "description": "Damped nonlinear oscillator acceleration",
        "required_files": ["train.csv", "test_id.csv", "test_ood.csv"],
    },
    "oscillator2": {
        "description": "Damped nonlinear oscillator (variant 2)",
        "required_files": ["train.csv", "test_id.csv", "test_ood.csv"],
    },
    "stressstrain": {
        "description": "Stress prediction in Aluminium rod",
        "required_files": ["train.csv", "test_id.csv", "test_ood.csv"],
    }
}

# Dataset category configuration
DATASET_CATEGORIES = {
    "scientific_regression": {
        "datasets": SCIENTIFIC_REGRESSION_DATASETS,
        "release_filename": "scientific_regression.zip",
        "release_url": f"https://github.com/{GITHUB_REPO}/releases/download/data-{DATA_VERSION}/scientific_regression.zip"
    }
}


def get_dataset_metadata(category: str, dataset_name: str) -> Dict:
    """
    Get metadata for a specific dataset.

    Args:
        category: Dataset category (e.g., "scientific_regression")
        dataset_name: Name of the dataset (e.g., "bactgrow")

    Returns:
        Dictionary containing dataset metadata

    Raises:
        ValueError: If category or dataset is unknown
    """
    if category not in DATASET_CATEGORIES:
        raise ValueError(
            f"Unknown dataset category: {category}. "
            f"Available: {list(DATASET_CATEGORIES.keys())}"
        )

    category_config = DATASET_CATEGORIES[category]
    datasets = category_config["datasets"]

    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown dataset: {dataset_name} in category {category}. "
            f"Available: {list(datasets.keys())}"
        )

    return datasets[dataset_name]


def get_release_url(category: str) -> str:
    """
    Get GitHub release download URL for a dataset category.

    Args:
        category: Dataset category (e.g., "scientific_regression")

    Returns:
        Download URL string

    Raises:
        ValueError: If category is unknown
    """
    if category not in DATASET_CATEGORIES:
        raise ValueError(
            f"Unknown dataset category: {category}. "
            f"Available: {list(DATASET_CATEGORIES.keys())}"
        )

    return DATASET_CATEGORIES[category]["release_url"]


def get_required_files(category: str, dataset_name: str) -> List[str]:
    """
    Get list of required files for dataset verification.

    Args:
        category: Dataset category
        dataset_name: Name of the dataset

    Returns:
        List of required filenames
    """
    metadata = get_dataset_metadata(category, dataset_name)
    return metadata["required_files"]
