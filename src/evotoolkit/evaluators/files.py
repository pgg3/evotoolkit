"""File-set validation shared by workspace writers."""

import re
from pathlib import Path

from evotoolkit.core.types import Files


def file_path(root: Path, name: str) -> Path:
    """Reject traversal, platform aliases, and writes through outside symlinks."""
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or re.search(r'[<>:"|?*\x00-\x1f]', name)
        or name.startswith("/")
        or any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in name.split("/"))
        or any(re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", part, re.IGNORECASE) for part in name.split("/"))
    ):
        raise ValueError(f"Unsafe artifact path: {name!r}")
    path = root / name
    current = root
    for component in name.split("/"):
        current /= component
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError(f"Artifact path contains a link: {name!r}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes workspace: {name!r}")
    return path


def write_files(root: Path, files: Files) -> None:
    paths = [(file_path(root, name), content) for name, content in files.items()]
    if len({path for path, _ in paths}) != len(paths):
        raise ValueError("Artifact paths alias the same file")
    if any(not isinstance(content, str) for _, content in paths):
        raise ValueError("Artifact contents must be strings")
    for path, content in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
