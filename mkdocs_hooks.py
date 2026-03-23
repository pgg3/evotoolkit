from pathlib import Path
from shutil import copy2


def on_post_build(config, **kwargs):
    """Mirror the root sitemap into translated subdirectories for RTD addons."""
    i18n_plugin = config.plugins.get("i18n")
    if i18n_plugin is None:
        return

    site_dir = Path(config.site_dir)
    sitemap_files = [site_dir / "sitemap.xml", site_dir / "sitemap.xml.gz"]
    if not any(path.exists() for path in sitemap_files):
        return

    for language in i18n_plugin.config.languages:
        if not language.build or language.default or language.locale == "null":
            continue

        target_dir = site_dir / str(language.locale)
        if not target_dir.exists():
            continue

        for source in sitemap_files:
            if source.exists():
                copy2(source, target_dir / source.name)
