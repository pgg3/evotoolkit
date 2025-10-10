# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Dataset downloader from GitHub releases.
"""

import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from .constants import get_release_url, get_required_files


class DownloadError(Exception):
    """Raised when dataset download fails."""

    pass


def download_with_progress(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """
    Download file from URL with progress indication.

    Args:
        url: Download URL
        dest_path: Destination file path
        chunk_size: Download chunk size in bytes

    Raises:
        DownloadError: If download fails
    """
    try:
        # Create request with timeout
        req = urllib.request.Request(url, headers={"User-Agent": "EvoToolkit/1.0"})

        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get("content-length", 0))

            # Download to temporary file first
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = dest_path.with_suffix(".tmp")

            try:
                with open(temp_path, "wb") as f:
                    downloaded = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Print progress
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownloading: {progress:.1f}%", end="", flush=True)

                print()  # New line after progress

                # Move to final location
                if temp_path.exists():
                    shutil.move(str(temp_path), str(dest_path))

            finally:
                # Cleanup temp file if exists
                if temp_path.exists():
                    temp_path.unlink()

    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise DownloadError(
                f"Dataset not found at {url}. "
                f"Please ensure the data release exists on GitHub."
            ) from e
        else:
            raise DownloadError(f"HTTP error {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise DownloadError(f"Network error: {e.reason}") from e
    except Exception as e:
        raise DownloadError(f"Download failed: {str(e)}") from e


def verify_dataset(category: str, dataset_name: str, dataset_path: Path) -> bool:
    """
    Verify that dataset has all required files.

    Args:
        category: Dataset category
        dataset_name: Name of the dataset
        dataset_path: Path to dataset directory

    Returns:
        True if all required files exist
    """
    if not dataset_path.exists():
        return False

    required_files = get_required_files(category, dataset_name)
    for filename in required_files:
        if not (dataset_path / filename).exists():
            return False

    return True


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extract zip file to destination.

    Args:
        zip_path: Path to zip file
        extract_to: Destination directory

    Raises:
        DownloadError: If extraction fails
    """
    try:
        print(f"Extracting to {extract_to}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    except zipfile.BadZipFile as e:
        raise DownloadError("Downloaded file is corrupted. Please try again.") from e
    except Exception as e:
        raise DownloadError(f"Extraction failed: {str(e)}") from e


def download_dataset_category(category: str, target_dir: Path) -> Path:
    """
    Download entire dataset category from GitHub release.

    Args:
        category: Dataset category (e.g., "scientific_regression")
        target_dir: Target directory to extract datasets

    Returns:
        Path to the extracted dataset category directory

    Raises:
        DownloadError: If download or extraction fails
    """
    # Get download URL
    url = get_release_url(category)

    # Create temp directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_path = temp_path / f"{category}.zip"

        # Download
        print(f"Downloading {category} datasets from GitHub release...")
        print(f"URL: {url}")
        download_with_progress(url, zip_path)

        # Extract
        target_dir.mkdir(parents=True, exist_ok=True)
        extract_zip(zip_path, target_dir)

    return target_dir


def ensure_dataset_downloaded(
    category: str, dataset_name: str, data_dir: Optional[Path] = None
) -> Path:
    """
    Ensure dataset is downloaded and available.

    Downloads the entire category if needed, then returns path to specific dataset.

    Args:
        category: Dataset category (e.g., "scientific_regression")
        dataset_name: Name of the dataset (e.g., "bactgrow")
        data_dir: Custom data directory (defaults to ~/.evotool/data/)

    Returns:
        Path to the specific dataset directory

    Raises:
        DownloadError: If download fails
        FileNotFoundError: If dataset not found after download
    """
    # Determine data directory
    if data_dir is None:
        data_dir = Path.home() / ".evotool" / "data"
    else:
        data_dir = Path(data_dir)

    # Category base directory
    category_dir = data_dir / category
    dataset_path = category_dir / dataset_name

    # Check if dataset already exists and is valid
    if verify_dataset(category, dataset_name, dataset_path):
        return dataset_path

    # Need to download - check if we already have category but missing specific dataset
    if category_dir.exists():
        print(f"Dataset '{dataset_name}' not found in existing {category} directory.")
        print(
            "This might indicate incomplete download. Re-downloading entire category..."
        )
        # Clean up existing directory to ensure fresh download
        shutil.rmtree(category_dir)

    # Download entire category
    try:
        download_dataset_category(category, category_dir)
    except DownloadError as e:
        raise DownloadError(
            f"Failed to download {category} datasets: {str(e)}\n\n"
            f"Troubleshooting:\n"
            f"  1. Check your internet connection\n"
            f"  2. Verify the release exists: https://github.com/pgg3/evotoolkit/releases\n"
            f"  3. Try again later if GitHub is experiencing issues\n"
        ) from e

    # Verify dataset exists after download
    if not verify_dataset(category, dataset_name, dataset_path):
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found after downloading {category}. "
            f"This might be a bug - please report it at: "
            f"https://github.com/pgg3/evotoolkit/issues"
        )

    print(f"✓ Dataset '{dataset_name}' ready at: {dataset_path}")
    return dataset_path
