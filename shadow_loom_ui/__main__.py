"""Allow running as: python -m shadow_loom_ui"""
from shadow_loom_ui.app import config, ui, _ui_settings  # noqa: F401

ui.run(
    host=_ui_settings.host,
    port=_ui_settings.port,
    title=_ui_settings.title,
    storage_secret=config.STORAGE_SECRET,
    dark=_ui_settings.dark_mode,
    reload=_ui_settings.reload,
)
