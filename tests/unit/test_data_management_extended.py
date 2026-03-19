# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Additional data module and downloader edge-case tests."""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

import evotoolkit.data as data_module
import evotoolkit.data.downloader as downloader_module
from evotoolkit.data.downloader import DownloadError


class BrokenResponse:
    headers = {"content-length": "3"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, chunk_size: int = -1):
        raise OSError("stream broken")


class TestDataModuleExtended:
    def test_default_data_dir_prefers_new_directory_when_present(self, tmp_path, monkeypatch):
        home_dir = tmp_path / "home"
        new_dir = home_dir / ".evotoolkit" / "data"
        new_dir.mkdir(parents=True)
        monkeypatch.setattr(data_module.Path, "home", lambda: home_dir)

        assert data_module._default_data_dir() == new_dir

    def test_get_dataset_path_converts_string_directory(self, tmp_path, monkeypatch):
        calls = []

        def fake_ensure(category, dataset_name, base_dir):
            calls.append(Path(base_dir))
            target = Path(base_dir) / category / dataset_name
            target.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(data_module, "ensure_dataset_downloaded", fake_ensure)

        result = data_module.get_dataset_path("scientific_regression", data_dir=str(tmp_path))

        assert result == tmp_path / "scientific_regression"
        assert calls == [tmp_path]


class TestDownloaderHelpersExtended:
    def test_download_with_progress_wraps_http_404(self, tmp_path, monkeypatch):
        def raise_404(req, timeout=30):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

        monkeypatch.setattr(downloader_module.urllib.request, "urlopen", raise_404)

        with pytest.raises(DownloadError, match="Dataset not found"):
            downloader_module.download_with_progress("https://example.com/data.zip", tmp_path / "file.zip")

    def test_download_with_progress_wraps_url_errors(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            downloader_module.urllib.request,
            "urlopen",
            lambda req, timeout=30: (_ for _ in ()).throw(urllib.error.URLError("offline")),
        )

        with pytest.raises(DownloadError, match="Network error"):
            downloader_module.download_with_progress("https://example.com/data.zip", tmp_path / "file.zip")

    def test_download_with_progress_cleans_up_temp_files_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(downloader_module.urllib.request, "urlopen", lambda req, timeout=30: BrokenResponse())
        dest_path = tmp_path / "download.zip"

        with pytest.raises(DownloadError, match="Download failed"):
            downloader_module.download_with_progress("https://example.com/data.zip", dest_path)

        assert not dest_path.with_suffix(".tmp").exists()

    def test_extract_zip_reports_bad_archives(self, tmp_path):
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_text("not-a-zip", encoding="utf-8")

        with pytest.raises(DownloadError, match="corrupted"):
            downloader_module.extract_zip(bad_zip, tmp_path / "out")

    def test_extract_zip_wraps_unexpected_errors(self, tmp_path, monkeypatch):
        zip_path = tmp_path / "good.zip"
        zip_path.write_bytes(b"placeholder")

        class FakeZipFile:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                raise RuntimeError("boom")

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(downloader_module.zipfile, "ZipFile", FakeZipFile)

        with pytest.raises(DownloadError, match="Extraction failed"):
            downloader_module.extract_zip(zip_path, tmp_path / "out")
