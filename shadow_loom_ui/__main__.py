"""Allow running as: python -m shadow_loom_ui"""
from shadow_loom_ui.app import config, ui  # noqa: F401

ui.run(
    host="0.0.0.0",
    port=7860,
    title="Shadow Loom",
    storage_secret=config.STORAGE_SECRET,
    dark=True,
    reload=False,
)
