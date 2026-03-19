# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for dataset metadata and download helpers."""

import io
import zipfile
from pathlib import Path

import pytest

import evotoolkit.data as data_module
import evotoolkit.data.downloader as downloader_module
from evotoolkit.data.constants import (
    DATASET_CATEGORIES,
    get_dataset_metadata,
    get_release_url,
    get_required_files,
)
from evotoolkit.data.downloader import DownloadError


class FakeDownloadResponse:
    def __init__(self, content: bytes):
        self._buffer = io.BytesIO(content)
        self.headers = {"content-length": str(len(content))}

    def read(self, chunk_size: int = -1):
        return self._buffer.read(chunk_size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestDataConstants:
    def test_dataset_metadata_and_required_files(self):
        metadata = get_dataset_metadata("scientific_regression", "bactgrow")

        assert metadata["description"] == "E. Coli bacterial growth rate prediction"
        assert get_required_files("scientific_regression", "bactgrow") == ["train.csv", "test_id.csv", "test_ood.csv"]
        assert get_release_url("scientific_regression") == DATASET_CATEGORIES["scientific_regression"]["release_url"]

    def test_invalid_dataset_metadata_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset category"):
            get_dataset_metadata("missing", "bactgrow")

        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_metadata("scientific_regression", "missing")


class TestDataModule:
    def test_list_available_datasets_returns_copy(self):
        datasets = data_module.list_available_datasets("scientific_regression")
        datasets["new_dataset"] = {}

        assert "new_dataset" not in DATASET_CATEGORIES["scientific_regression"]["datasets"]

    def test_get_dataset_path_returns_existing_directory(self, tmp_path):
        category_dir = tmp_path / "scientific_regression"
        category_dir.mkdir()
        (category_dir / "bactgrow").mkdir()

        result = data_module.get_dataset_path("scientific_regression", data_dir=tmp_path)

        assert result == category_dir

    def test_get_dataset_path_triggers_download_when_missing(self, tmp_path, monkeypatch):
        calls = []

        def fake_ensure(category, dataset_name, base_dir):
            calls.append((category, dataset_name, Path(base_dir)))
            target = Path(base_dir) / category / dataset_name
            target.mkdir(parents=True)

        monkeypatch.setattr(data_module, "ensure_dataset_downloaded", fake_ensure)

        result = data_module.get_dataset_path("scientific_regression", data_dir=tmp_path)

        assert result == tmp_path / "scientific_regression"
        assert calls == [("scientific_regression", "bactgrow", tmp_path)]

    def test_get_dataset_path_rejects_unknown_category(self):
        with pytest.raises(ValueError, match="Unknown dataset category"):
            data_module.get_dataset_path("missing")

    def test_default_data_dir_prefers_legacy_path(self, tmp_path, monkeypatch):
        home_dir = tmp_path / "home"
        legacy_dir = home_dir / ".evotool" / "data"
        legacy_dir.mkdir(parents=True)
        monkeypatch.setattr(data_module.Path, "home", lambda: home_dir)

        assert data_module._default_data_dir() == legacy_dir


class TestDownloaderHelpers:
    def test_verify_dataset(self, tmp_path):
        dataset_path = tmp_path / "scientific_regression" / "bactgrow"
        dataset_path.mkdir(parents=True)

        assert downloader_module.verify_dataset("scientific_regression", "bactgrow", dataset_path) is False

        for filename in get_required_files("scientific_regression", "bactgrow"):
            (dataset_path / filename).write_text("ok", encoding="utf-8")

        assert downloader_module.verify_dataset("scientific_regression", "bactgrow", dataset_path) is True

    def test_download_with_progress_writes_destination_file(self, tmp_path, monkeypatch):
        content = b"abc123"
        monkeypatch.setattr(
            downloader_module.urllib.request,
            "urlopen",
            lambda req, timeout=30: FakeDownloadResponse(content),
        )

        dest_path = tmp_path / "download.zip"
        downloader_module.download_with_progress("https://example.com/file.zip", dest_path, chunk_size=2)

        assert dest_path.read_bytes() == content

    def test_extract_zip_extracts_contents(self, tmp_path):
        zip_path = tmp_path / "sample.zip"
        extract_to = tmp_path / "out"

        with zipfile.ZipFile(zip_path, "w") as zip_file:
            zip_file.writestr("scientific_regression/bactgrow/train.csv", "x,y\n1,2\n")

        downloader_module.extract_zip(zip_path, extract_to)

        assert (extract_to / "scientific_regression" / "bactgrow" / "train.csv").exists()

    def test_ensure_dataset_downloaded_returns_existing_valid_dataset(self, tmp_path):
        dataset_path = tmp_path / "scientific_regression" / "bactgrow"
        dataset_path.mkdir(parents=True)
        for filename in get_required_files("scientific_regression", "bactgrow"):
            (dataset_path / filename).write_text("ok", encoding="utf-8")

        result = downloader_module.ensure_dataset_downloaded("scientific_regression", "bactgrow", data_dir=tmp_path)

        assert result == dataset_path

    def test_ensure_dataset_downloaded_redownloads_incomplete_category(self, tmp_path, monkeypatch):
        category_dir = tmp_path / "scientific_regression"
        incomplete_dataset = category_dir / "bactgrow"
        incomplete_dataset.mkdir(parents=True)
        (incomplete_dataset / "train.csv").write_text("partial", encoding="utf-8")

        calls = []

        def fake_download(category, target_dir):
            calls.append((category, Path(target_dir)))
            dataset_dir = Path(target_dir) / "bactgrow"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            for filename in get_required_files(category, "bactgrow"):
                (dataset_dir / filename).write_text("ok", encoding="utf-8")
            return target_dir

        monkeypatch.setattr(downloader_module, "download_dataset_category", fake_download)

        result = downloader_module.ensure_dataset_downloaded("scientific_regression", "bactgrow", data_dir=tmp_path)

        assert calls == [("scientific_regression", category_dir)]
        assert result == category_dir / "bactgrow"

    def test_ensure_dataset_downloaded_wraps_download_errors(self, tmp_path, monkeypatch):
        def failing_download(category, target_dir):
            raise DownloadError("boom")

        monkeypatch.setattr(downloader_module, "download_dataset_category", failing_download)

        with pytest.raises(DownloadError, match="Troubleshooting"):
            downloader_module.ensure_dataset_downloaded("scientific_regression", "bactgrow", data_dir=tmp_path)

    def test_ensure_dataset_downloaded_raises_if_dataset_missing_after_download(self, tmp_path, monkeypatch):
        monkeypatch.setattr(downloader_module, "download_dataset_category", lambda category, target_dir: target_dir)

        with pytest.raises(FileNotFoundError, match="not found after downloading"):
            downloader_module.ensure_dataset_downloaded("scientific_regression", "bactgrow", data_dir=tmp_path)
