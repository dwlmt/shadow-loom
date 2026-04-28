"""Shadow Loom UI theme — palette, typography, icons.

Single source of truth for the modern visual language across the app.

- Light theme by default; dark mode toggle preserved (driven by
  ``settings.ui.dark_mode``).
- Inter font loaded from Google Fonts.
- Feather icons loaded via CDN; ``feather(name)`` returns a NiceGUI element
  emitting an ``<i data-feather="name">`` tag, plus a hook that calls
  ``feather.replace()`` after each NiceGUI client update so newly added
  icons get rendered.
- Warm "mort artistic" palette (copper / slate / aged gold) drives both the
  app shell and the chart series colors.
"""

from __future__ import annotations

from nicegui import ui

# =====================================================================
# Palette constants
# =====================================================================

# Brand
PRIMARY = "#C68661"        # Copper
SECONDARY = "#5C7C8A"      # Slate Blue
ACCENT = "#D4A35B"         # Aged Gold

# 10-color "mort artistic" chart palette (also surfaces semantic colors)
CHART_COLORS: list[str] = [
    "#C68661",   # 1. Copper
    "#5C7C8A",   # 2. Slate Blue
    "#D4A35B",   # 3. Aged Gold
    "#456A6B",   # 4. Deep Spruce
    "#B58988",   # 5. Dusty Rose
    "#95A577",   # 6. Olive
    "#856B7D",   # 7. Faded Plum
    "#8C7A6B",   # 8. Warm Clay
    "#9E4D4D",   # 9. Dusty Brick
    "#1F2731",   # 10. Ink
]

# Aliases for semantic use
COPPER = CHART_COLORS[0]
SLATE_BLUE = CHART_COLORS[1]
AGED_GOLD = CHART_COLORS[2]
SPRUCE = CHART_COLORS[3]
ROSE = CHART_COLORS[4]
OLIVE = CHART_COLORS[5]
PLUM = CHART_COLORS[6]
CLAY = CHART_COLORS[7]
BRICK = CHART_COLORS[8]
INK = CHART_COLORS[9]

# Neutral scale
SLATE_50 = "#f8fafc"
SLATE_100 = "#f1f5f9"
SLATE_200 = "#e2e8f0"
SLATE_300 = "#cbd5e1"
SLATE_400 = "#94a3b8"
SLATE_500 = "#64748b"
SLATE_600 = "#475569"
SLATE_700 = "#334155"
SLATE_800 = "#1e293b"
SLATE_900 = "#0f172a"

# Semantic statuses (used for badges, gauges, alerts)
POSITIVE = OLIVE
WARNING = AGED_GOLD
NEGATIVE = BRICK
INFO = SLATE_BLUE


# =====================================================================
# Reusable Tailwind/Quasar class strings
# =====================================================================

CARD_CLS = (
    "bg-white border border-slate-200 rounded-xl shadow-sm p-6"
)
CARD_TIGHT_CLS = (
    "bg-white border border-slate-200 rounded-xl shadow-sm p-4"
)
CARD_HEADER_CLS = (
    "w-full p-4 border-b border-slate-200 bg-slate-50/50"
)
SECTION_TITLE_CLS = "text-lg font-semibold text-slate-800"
PAGE_TITLE_CLS = "text-3xl font-bold text-slate-800"
SUBTITLE_CLS = "text-sm text-slate-500"
BTN_PRIMARY_CLS = "rounded-lg shadow-sm"


# =====================================================================
# Head HTML — fonts, Feather icons, base CSS, ECharts color reuse
# =====================================================================

def _head_html() -> str:
    palette_css = ", ".join(f"'{c}'" for c in CHART_COLORS)
    # Fonts are loaded with the print/onload pattern so they NEVER block
    # the initial render.  If the CDN is unreachable the page still
    # appears immediately with the system-font fallback; when the font
    # arrives the swap is invisible.  This is critical for:
    #   - sandboxed/offline environments where fonts.googleapis.com is
    #     blocked or slow,
    #   - browsers that synchronously block first paint on font CSS.
    return f"""
<link rel="preconnect" href="https://fonts.googleapis.com" crossorigin>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="preload" as="style"
  href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
  onload="this.onload=null;this.rel='stylesheet'">
<link rel="preload" as="style"
  href="https://fonts.googleapis.com/icon?family=Material+Icons+Outlined"
  onload="this.onload=null;this.rel='stylesheet'">
<noscript>
  <link rel="stylesheet"
    href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
  <link rel="stylesheet"
    href="https://fonts.googleapis.com/icon?family=Material+Icons+Outlined">
</noscript>
<style>
  :root {{
    --sl-primary: {PRIMARY};
    --sl-secondary: {SECONDARY};
    --sl-accent: {ACCENT};
    --sl-bg: {SLATE_50};
    --sl-surface: #ffffff;
    --sl-text: {SLATE_800};
    --sl-muted: {SLATE_500};
    --sl-border: {SLATE_200};
  }}
  body, .q-page, .nicegui-content {{
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto,
      'Helvetica Neue', Arial, sans-serif;
    background-color: var(--sl-bg);
    color: var(--sl-text);
  }}
  .body--dark, body.body--dark, .body--dark .q-page, .body--dark .nicegui-content {{
    background-color: {INK};
    color: {SLATE_100};
  }}
  /* Feather icon defaults — kept for any legacy callers that still
     emit raw <i data-feather> tags; current ``feather()`` helper now
     renders native ``ui.icon`` (Material Outlined) instead. */
  .feather {{
    width: 18px;
    height: 18px;
    stroke-width: 2;
    vertical-align: middle;
  }}
  .feather-lg {{ width: 24px; height: 24px; }}
  .feather-xl {{ width: 32px; height: 32px; }}
  .feather-sm {{ width: 14px; height: 14px; }}
  /* Force Quasar's Material Icons font slot to use the Outlined variant
     so all existing widget icons (icon="save"/icon="bolt"/...) render
     thin-stroked.  Falls back to the regular Material Icons font (which
     Quasar bundles) so icons remain visible even when the outlined CDN
     font hasn't loaded yet. */
  .material-icons,
  .q-icon.material-icons,
  i.material-icons {{
    font-family: 'Material Icons Outlined', 'Material Icons' !important;
    font-weight: 400 !important;
  }}
  /* Subtle hover for interactive cards */
  .sl-card-hover {{
    transition: box-shadow 120ms ease, transform 120ms ease, border-color 120ms ease;
  }}
  .sl-card-hover:hover {{
    border-color: {SLATE_300};
    box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.08);
    transform: translateY(-1px);
  }}
  /* Chart palette CSS variable for ECharts consumers */
  :root {{ --sl-chart-palette: {palette_css}; }}
</style>
"""


# =====================================================================
# Public API
# =====================================================================

_HEAD_INJECTED = False


def apply_theme(dark: bool = False) -> None:
    """Apply the Shadow Loom theme to the current page.

    Call once per ``@ui.page`` handler — replaces any prior
    ``ui.dark_mode(...)`` + ``ui.colors(...)`` setup.
    """
    ui.dark_mode(dark)
    ui.colors(
        primary=PRIMARY,
        secondary=SECONDARY,
        accent=ACCENT,
        positive=POSITIVE,
        negative=NEGATIVE,
        warning=WARNING,
        info=INFO,
    )
    ui.add_head_html(_head_html())
    set_chart_dark(dark)


# =====================================================================
# Feather icon helper
# =====================================================================

# 1-to-1 mapping from the legacy Material/Quasar names to Feather names.
# Anything not in the map is passed through (Feather has many same-name icons).
_FEATHER_ALIASES: dict[str, str] = {
    # navigation / shell
    "menu": "menu",
    "arrow_back": "arrow-left",
    "arrow_forward": "arrow-right",
    "settings": "settings",
    "logout": "log-out",
    "login": "log-in",
    "account_circle": "user",
    "person_outline": "user",
    "person_add": "user-plus",
    # branding / story
    "auto_stories": "book-open",
    "menu_book": "book",
    "book": "book",
    "description": "file-text",
    "article": "file-text",
    "edit_note": "edit",
    # workspace tabs
    "hub": "share-2",
    "device_hub": "git-branch",
    "fact_check": "check-square",
    "ios_share": "upload",
    "share": "share-2",
    # actions
    "save": "save",
    "add": "plus",
    "delete": "trash-2",
    "content_copy": "copy",
    "download": "download",
    "upload": "upload",
    "refresh": "refresh-cw",
    "refresh-ccw": "refresh-ccw",
    "play_arrow": "play",
    "replay": "refresh-ccw",
    "close": "x",
    "search": "search",
    # status / state
    "star": "star",
    "star_outline": "star",
    "warning": "alert-triangle",
    "error": "x-circle",
    "error_outline": "alert-circle",
    "check_circle": "check-circle",
    "info": "info",
    "hourglass_top": "clock",
    "history": "clock",
    "pending_actions": "clipboard",
    "clear_all": "trash",
    "visibility_off": "eye-off",
    "visibility": "eye",
    # graph / topology
    "account_tree": "git-branch",
    "alt_route": "git-branch",
    "call_split": "git-branch",
    "trending_up": "trending-up",
    "data_object": "code",
    "code": "code",
    # places / people
    "public": "globe",
    "place": "map-pin",
    "map": "map",
    "people": "users",
    "users": "users",
    "user": "user",
    "category": "box",
    "box": "box",
    # comms
    "mail": "mail",
    "send": "send",
    "link": "link",
    "vpn_key": "key",
    # affect / psychology
    "psychology": "cpu",
    "favorite": "heart",
    "theater_comedy": "film",
    "science": "zap",
    "auto_fix_high": "zap",
    "bolt": "zap",
    "flash_on": "zap",
    # misc
    "folder_open": "folder",
    "folder": "folder",
}


def feather_name(name: str) -> str:
    """Legacy shim: translate a Material/Quasar icon name to a Feather name.

    Kept for backward-compatibility with any caller that imported it
    directly.  In the current implementation we render Material Icons
    Outlined natively (see :func:`feather`), so this function is only
    used by the inverse lookup in :data:`_MATERIAL_FROM_FEATHER`.
    """
    return _FEATHER_ALIASES.get(name, name.replace("_", "-"))


# Reverse map: feather-name -> material-icon ligature.  Built from
# ``_FEATHER_ALIASES`` so the two stay in sync; entries below it cover
# Feather-native names that callers passed directly.
_MATERIAL_FROM_FEATHER: dict[str, str] = {
    feather: material for material, feather in _FEATHER_ALIASES.items()
}
_MATERIAL_FROM_FEATHER.update({
    # Feather-native names callers pass directly that need a Material glyph.
    "arrow-left": "arrow_back",
    "arrow-right": "arrow_forward",
    "book-open": "auto_stories",
    "book": "menu_book",
    "log-out": "logout",
    "log-in": "login",
    "user": "person",
    "user-plus": "person_add",
    "plus": "add",
    "trash": "delete_outline",
    "trash-2": "delete",
    "copy": "content_copy",
    "check-square": "fact_check",
    "git-branch": "account_tree",
    "share-2": "share",
    "file-text": "description",
    "check-circle": "check_circle",
    "x-circle": "cancel",
    "x": "close",
    "alert-triangle": "warning",
    "alert-circle": "error_outline",
    "clock": "schedule",
    "clipboard": "assignment",
    "cpu": "memory",
    "film": "theaters",
    "zap": "bolt",
    "map-pin": "place",
    "users": "people",
    "box": "category",
    "globe": "public",
    "heart": "favorite",
    "eye": "visibility",
    "eye-off": "visibility_off",
    "play": "play_arrow",
    "refresh-cw": "refresh",
    "refresh-ccw": "replay",
    "trending-up": "trending_up",
    "send": "send",
    "save": "save",
    "search": "search",
    "settings": "settings",
    "edit": "edit",
    "code": "code",
    "info": "info",
    "star": "star",
    "menu": "menu",
    "mail": "email",
    "key": "vpn_key",
    "map": "map",
    "folder": "folder",
    "share": "share",
    "upload": "upload",
    "download": "download",
})


def feather(name: str, *, size: str = "", classes: str = "", color: str = "") -> ui.icon:
    """Render an icon.

    Despite the legacy name, this now emits a native Quasar
    ``ui.icon`` using the Material Icons Outlined font (which we load
    in :func:`_head_html` and alias as the default ``material-icons``
    family).  This visually matches Feather while staying inside Vue's
    virtual DOM — no MutationObserver, no Vue/Feather feedback loop.

    Accepts either a Material name (``"save"``) or a Feather name
    (``"book-open"``); Feather names are translated via
    :data:`_MATERIAL_FROM_FEATHER`.

    Parameters
    ----------
    name : str
        Icon name.
    size : str
        Optional Quasar size keyword (``"xs"``, ``"sm"``, ``"md"``,
        ``"lg"``, ``"xl"``) — passed straight through.
    classes : str
        Extra Tailwind/Quasar classes.
    color : str
        CSS color or Quasar palette name.  If it looks like a hex
        color, applied as inline ``style``; otherwise passed as the
        Quasar ``color`` prop.
    """
    material = _MATERIAL_FROM_FEATHER.get(name, name.replace("-", "_"))
    icon = ui.icon(material)
    if size:
        icon.props(f"size={size}")
    if color:
        if color.startswith("#") or color.startswith("rgb"):
            icon.style(f"color: {color}")
        else:
            icon.props(f"color={color}")
    if classes:
        icon.classes(classes)
    return icon


# =====================================================================
# Chart theme accessor
# =====================================================================

# Per-process chart theme flag — viz.py reads this to pick light vs dark
# axis/grid/text colors.  Updated in :func:`apply_theme`.
_CHART_DARK = False


def set_chart_dark(dark: bool) -> None:
    global _CHART_DARK
    _CHART_DARK = dark
    # Refresh viz.py module-level palette constants so subsequent
    # renders pick up the new theme.  Imported lazily to avoid a
    # circular import at module load time.
    try:
        from shadow_loom_ui import viz as _viz
        _viz.apply_chart_theme()
    except Exception:  # pragma: no cover - viz import optional in tests
        pass


def chart_theme() -> dict:
    """Return chart theme tokens for ECharts renderers."""
    if _CHART_DARK:
        return {
            "bg": INK,
            "text": SLATE_100,
            "muted_text": SLATE_400,
            "grid": SLATE_700,
            "axis": SLATE_600,
            "tooltip_bg": "#2a3240",
            "tooltip_border": SLATE_600,
            "tooltip_text": SLATE_100,
            "palette": CHART_COLORS,
        }
    return {
        "bg": "#ffffff",
        "text": SLATE_800,
        "muted_text": SLATE_500,
        "grid": SLATE_200,
        "axis": SLATE_300,
        "tooltip_bg": "#ffffff",
        "tooltip_border": SLATE_200,
        "tooltip_text": SLATE_800,
        "palette": CHART_COLORS,
    }
