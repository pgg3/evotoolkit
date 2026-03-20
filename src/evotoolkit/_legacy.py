# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""Helpers for intentionally removed package surfaces."""


def raise_split_import_error(old_path: str, replacement: str) -> None:
    raise ModuleNotFoundError(
        f"'{old_path}' was removed from evotoolkit 3.0.0. "
        f"Install 'evotoolkit-tasks' and import from '{replacement}' instead."
    )
