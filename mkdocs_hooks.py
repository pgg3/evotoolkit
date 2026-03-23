import atexit
from pathlib import Path
from shutil import copy2

_registered = False
_site_dir: Path | None = None


def _mirror_sitemaps() -> None:
    if _site_dir is None:
        return

    site_dir = _site_dir
    sitemap_files = [site_dir / "sitemap.xml", site_dir / "sitemap.xml.gz"]
    if not any(path.exists() for path in sitemap_files):
        return

    target_dirs = {site_dir}
    target_dirs.update(index.parent for index in site_dir.rglob("index.html"))

    for target_dir in target_dirs:
        for source in sitemap_files:
            if not source.exists():
                continue

            destination = target_dir / source.name
            if destination.resolve() == source.resolve():
                continue

            copy2(source, destination)


def on_post_build(config, **kwargs):
    """Mirror the root sitemap into every page directory for RTD addons."""
    global _registered, _site_dir

    _site_dir = Path(config.site_dir)
    if not _registered:
        atexit.register(_mirror_sitemaps)
        _registered = True
