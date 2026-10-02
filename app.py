"""
MTA Subway Ridership Analytics
==============================
Phase 1 - exploratory / temporal / station / geographic analysis for a
time-series project built on *MTA Subway Hourly Ridership: 2020-2024*
(Socrata view ``wujg-7c2s``).

Run with:  streamlit run app.py

The raw dataset holds ~121 million rows. Nothing in this app ever reads it:
every figure is built from a *server-side aggregate* cube (see
``utils/data_loader.py``) that is memoised to Parquet under
``data/processed/`` and further cached with ``@st.cache_data``.

This dashboard is deliberately descriptive only. It contains no forecasting
model, no train/test split and no accuracy metrics - those belong to later
phases. Every number shown is computed at render time; nothing is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an
from utils import data_loader as dl
from utils import preprocessing as pp
from utils.data_loader import ApiUnavailableError, DataLoadError, Filters
from utils.preprocessing import DOW_NUMBER_TO_NAME

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

APP_TITLE = "MTA Subway Ridership Analytics"
APP_SUBTITLE = "Temporal, Station & Geographic Analysis — 2020–2024"
DATASET_LABEL = "MTA Subway Hourly Ridership: 2020–2024"

#: Above this many hours an hourly axis carries more points than anyone can
#: read, so long windows step up to a coarser grain automatically.
HOURLY_POINT_BUDGET = 6000

#: Publication-neutral colour per borough; an unseen borough falls back to grey.
BOROUGH_COLORS = {
    "Manhattan": "#D1495B",
    "Brooklyn": "#1D6996",
    "Queens": "#2A9D8F",
    "Bronx": "#EDAE49",
    "Staten Island": "#8E7DBE",
}
BOROUGH_FALLBACK = "#8A94A6"

# --------------------------------------------------------------------------
# Theme
#
# The presentation layer below is carried over from the previous dashboard
# build so the app keeps the same restrained academic look: one primary, one
# accent, a neutral grid, expressed once per colour scheme. `P` is the
# *active* palette, re-bound at the start of every run by
# `configure_palette()` so the CSS and every Plotly figure stay in step with
# the light/dark toggle.
# --------------------------------------------------------------------------

THEMES: dict = {
    "light": {
        "primary": "#2F5597",
        "accent": "#C55A11",
        "grid": "#E5E7EB",
        "text": "#111827",
        "body": "#374151",
        "muted": "#6B7280",
        "faint": "#9CA3AF",
        "bg": "#FFFFFF",
        "paper": "#FFFFFF",
        "panel": "#F9FAFB",
        "kpi_bg": "#FFFFFF",
        "kpi_border": "#E5E7EB",
        "kpi_label": "#4B5563",
        "strip_border": "#E5E7EB",
        "warn_border": "#F3D19E",
        "warn_bg": "#FDF8F0",
        "warn_strong": "#8A3E08",
        "heat_scale": "YlOrRd",
        "heat_reverse": False,
        "hover_bg": "#FFFFFF",
        "hover_border": "#D1D5DB",
    },
    "dark": {
        # Lifted blues/oranges so marks stay legible on a dark surface.
        "primary": "#6FA8DC",
        "accent": "#F0A868",
        "grid": "#333A45",
        "text": "#F3F4F6",
        "body": "#D1D5DB",
        "muted": "#9CA3AF",
        "faint": "#6B7280",
        "bg": "#14181F",
        "paper": "#14181F",
        "panel": "#1C222B",
        "kpi_bg": "#1C222B",
        "kpi_border": "#2B333F",
        "kpi_label": "#9CA3AF",
        "strip_border": "#2B333F",
        "warn_border": "#6B4A21",
        "warn_bg": "#2A2117",
        "warn_strong": "#F0C08A",
        # Reversed so low values recede into the background and peaks glow.
        "heat_scale": "YlOrRd",
        "heat_reverse": True,
        "hover_bg": "#1C222B",
        "hover_border": "#3A4452",
    },
}

#: Active palette, re-bound on every run by :func:`configure_palette`.
P: dict = THEMES["light"]


def current_theme() -> str:
    """Active colour scheme, read from Streamlit's runtime theme.

    Falls back to the user's stored preference and finally to light.
    """
    preference = st.session_state.get("dark_mode")
    if preference is not None:
        return "dark" if preference else "light"
    active = getattr(st.context.theme, "type", None)
    return "dark" if active == "dark" else "light"


def configure_palette() -> str:
    """Bind the active palette and inject its CSS.

    Called once at the top of every run so the toggle, the CSS and every
    Plotly figure can never disagree about which scheme is showing.

    NOTE: Streamlit has no supported Python-side theme switch. ``st.context.theme``
    is a read-only snapshot inferred from the app background (upstream issue
    #11920), so assigning to it silently does nothing. Dark mode is therefore
    done purely in CSS (see ``css_blocks``).
    """
    global P
    theme = current_theme()
    P = THEMES[theme]
    for block in css_blocks(P):
        st.markdown(block, unsafe_allow_html=True)
    return theme


PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

# Streamlit silently truncates long HTML payloads passed to ``st.markdown``:
# a single ~6.5 kB dark stylesheet only delivered its first ~2.3 kB, so
# everything after that point was dropped without warning. Each stylesheet is
# therefore emitted as a small, self-contained block in its own
# ``st.markdown`` call, and :func:`css_blocks` refuses to emit an oversized one.

CSS_BLOCK_LIMIT = 1800  # comfortably below the observed truncation point


def _dark_blocks(p: dict) -> List[str]:
    """Stylesheets that re-colour Streamlit's own components for dark mode.

    Streamlit paints its widgets from generated emotion classes that cannot be
    themed from Python, so dark mode has to be done with selectors. These
    target stable ``data-testid`` hooks rather than emotion class names, which
    change between Streamlit releases.
    """
    if current_theme() != "dark":
        return []

    def block(body: str) -> str:
        return "<style>" + body + "</style>"

    return [
        # app shell + sidebar
        # NOTE: every shell selector is prefixed with `html body`. Streamlit's
        # own rules are more specific than a bare class/attribute selector, so
        # without the extra weight our `!important` loses the cascade and the
        # page stays white behind light text.
        block(f"""
  html body [data-testid="stApp"], html body .stApp {{
      background-color: {p['bg']} !important;
      color: {p['body']} !important;
  }}
  html body [data-testid="stHeader"] {{ background-color: transparent !important; }}
  html body [data-testid="stMain"], html body [data-testid="stMainBlockContainer"],
  html body [data-testid="stVerticalBlock"] {{
      background-color: transparent !important;
      color: {p['body']} !important;
  }}
  html body [data-testid="stSidebar"],
  html body [data-testid="stSidebarUserContent"],
  html body [data-testid="stSidebarContent"] {{
      background-color: {p['panel']} !important;
      color: {p['body']} !important;
  }}
  html body [data-testid="stSidebar"] * {{ color: {p['body']}; }}
  html body [data-testid="stSidebar"] h1,
  html body [data-testid="stSidebar"] h2,
  html body [data-testid="stSidebar"] h3 {{ color: {p['text']} !important; }}
"""),
        # typography
        block(f"""
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stMarkdownContainer"] strong {{
      color: {p['body']} !important;
  }}
  [data-testid="stMarkdownContainer"] code {{
      background-color: {p['panel']} !important;
      color: {p['accent']} !important;
  }}
  [data-testid="stCaptionContainer"] {{ color: {p['muted']} !important; }}
"""),
        # KPI cards
        block(f"""
  [data-testid="stMetric"] {{
      border: 1px solid {p['kpi_border']} !important;
      border-radius: 8px;
      background: {p['kpi_bg']} !important;
  }}
  [data-testid="stMetricValue"] {{ color: {p['text']} !important; }}
  [data-testid="stMetricLabel"] {{ color: {p['kpi_label']} !important; }}
  [data-testid="stMetricDelta"] {{
      background-color: rgba(255, 255, 255, 0.07) !important;
      color: {p['muted']} !important;
  }}
  [data-testid="stMetricDelta"] p,
  [data-testid="stMetricDelta"] span {{ color: {p['muted']} !important; }}
"""),
        # inputs + multiselect chips
        block(f"""
  [data-testid="stDateInputField"],
  [data-testid="stTextInput"] input,
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
      background-color: {p['kpi_bg']} !important;
      color: {p['text']} !important;
  }}
  [data-testid="stDateInputField"] input,
  [data-testid="stTextInput"] input,
  input::placeholder {{ color: {p['muted']} !important; }}
  [data-testid="stDateInputField"] input {{ color-scheme: dark; }}
  [data-testid="stMultiSelectTagsContainer"] > div > span,
  [data-testid="stMultiSelectTagsContainer"] span[title] {{
      background-color: {p['strip_border']} !important;
      color: {p['text']} !important;
  }}
  [data-testid="stMultiSelectTagsContainer"] svg path {{ fill: {p['muted']} !important; }}
"""),
        # buttons + toggles + sliders
        block(f"""
  [data-testid="stBaseButton-secondary"] {{
      background-color: {p['kpi_bg']} !important;
      color: {p['text']} !important;
      border: 1px solid {p['kpi_border']} !important;
  }}
  [data-testid="stBaseButton-secondary"]:hover {{
      background-color: {p['strip_border']} !important;
      color: {p['text']} !important;
      border-color: {p['primary']} !important;
  }}
  [data-testid="stCheckbox"] label,
  [data-testid="stRadio"] label,
  [role="radiogroup"] label {{ color: {p['text']} !important; }}
  [data-testid="stSlider"] [role="slider"] {{ background-color: {p['primary']} !important; }}
"""),
        # slider thumb value badge (drawn on a low-contrast primary bubble)
        block(f"""
  [data-testid="stSliderThumbValue"] {{
      background-color: {p['kpi_bg']} !important;
      color: {p['text']} !important;
      border: 1px solid {p['kpi_border']} !important;
      border-radius: 4px;
      padding: 0 4px;
  }}
  div:has([data-testid="stSliderThumbValue"]) {{
      background-color: transparent !important;
  }}
"""),
        # tabs + expanders
        block(f"""
  [data-testid="stTabs"] [role="tablist"] {{
      background-color: transparent !important;
      border-bottom: 1px solid {p['strip_border']} !important;
  }}
  [data-testid="stTab"] {{
      color: {p['muted']} !important;
      background-color: transparent !important;
  }}
  [data-testid="stTab"][aria-selected="true"] {{ color: {p['primary']} !important; }}
  [data-testid="stTabPanel"] {{ background-color: transparent !important; }}
  details[data-testid="stExpander"], [data-testid="stExpander"] {{
      background-color: {p['panel']} !important;
      border-color: {p['kpi_border']} !important;
  }}
  [data-testid="stExpander"] summary p {{ color: {p['text']} !important; }}
  [data-testid="stExpander"] svg path {{ fill: {p['muted']}; }}
"""),
        # alerts + dividers + element toolbars
        block(f"""
  [data-testid="stAlert"] {{ color: {p['body']} !important; }}
  [data-testid="stAlertContainer"] {{ border-color: {p['kpi_border']} !important; }}
  div[data-baseweb="notification"] {{ background-color: {p['panel']} !important; }}
  hr {{ border-color: {p['strip_border']} !important; }}
  [data-testid="stElementToolbar"],
  [data-testid="stElementToolbarContainer"],
  [data-testid="stElementToolbarButtonContainer"],
  [data-testid="stElementToolbarBody"] {{ background-color: transparent !important; }}
  [data-testid="stElementToolbar"] button {{ color: {p['muted']} !important; }}
"""),
        # popovers + dataframes
        block(f"""
  [data-baseweb="popover"] > div, [role="listbox"] {{
      background-color: {p['kpi_bg']} !important;
      color: {p['text']} !important;
  }}
  [role="option"] {{ color: {p['text']} !important; }}
  [role="option"]:hover {{ background-color: {p['strip_border']} !important; }}
  [data-testid="stDataFrame"] {{
      background-color: {p['kpi_bg']} !important;
      border: 1px solid {p['kpi_border']} !important;
      border-radius: 8px;
      padding: 2px;
  }}
"""),
        # scrollbars
        block(f"""
  ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
  ::-webkit-scrollbar-track {{ background: {p['panel']}; }}
  ::-webkit-scrollbar-thumb {{ background: {p['strip_border']}; border-radius: 5px; }}
  * {{ scrollbar-color: {p['strip_border']} {p['panel']}; }}
"""),
    ]


def _custom_blocks(p: dict) -> List[str]:
    """Stylesheets for the app's own markup, generated from the palette."""

    def block(body: str) -> str:
        return "<style>" + body + "</style>"

    return [
        block(f"""
  .block-container {{ padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1500px; }}
  .app-title {{
      font-size: 2.15rem; font-weight: 700; letter-spacing: -0.015em;
      line-height: 1.2; margin: 0 0 0.15rem 0; color: {p['text']};
  }}
  .app-subtitle {{
      font-size: 1.02rem; font-weight: 500; color: {p['body']};
      margin: 0 0 0.6rem 0;
  }}
  .app-eyebrow {{
      font-size: 0.72rem; font-weight: 700; letter-spacing: 0.13em;
      text-transform: uppercase; color: {p['muted']}; margin: 0 0 0.35rem 0;
  }}
"""),
        block(f"""
  .section-header {{
      font-size: 1.18rem; font-weight: 650; color: {p['text']};
      border-left: 4px solid {p['primary']}; padding: 0.12rem 0 0.12rem 0.6rem;
      margin: 0.1rem 0 0.15rem 0;
  }}
  .section-note {{ font-size: 0.82rem; color: {p['muted']}; margin: 0 0 0.9rem 0.55rem; }}
  .caption-note {{ font-size: 0.78rem; color: {p['muted']}; margin: 0.25rem 0 0 0; }}
"""),
        block(f"""
  .info-strip {{
      border: 1px solid {p['strip_border']}; border-left: 4px solid {p['primary']};
      border-radius: 6px; padding: 0.6rem 0.85rem; background: {p['panel']};
      font-size: 0.86rem; color: {p['body']}; margin: 0.5rem 0 0.9rem 0;
  }}
  .info-strip b {{ color: {p['text']}; }}
  .warn-strip {{
      border: 1px solid {p['warn_border']}; border-left: 4px solid {p['accent']};
      border-radius: 6px; padding: 0.6rem 0.85rem; background: {p['warn_bg']};
      font-size: 0.86rem; color: {p['body']}; margin: 0.5rem 0 0.9rem 0;
  }}
  .warn-strip b {{ color: {p['warn_strong']}; }}
"""),
        block(f"""
  .footer-note {{
      font-size: 0.76rem; color: {p['muted']}; text-align: center;
      border-top: 1px solid {p['strip_border']}; padding-top: 0.8rem; margin-top: 2.2rem;
  }}
  div[data-testid="stSidebar"] {{ border-right: 1px solid {p['strip_border']}; }}
"""),
        block("""
  [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"],
  [data-testid="stDecoration"], #MainMenu, footer, [data-testid="stFooter"] {
      display: none;
  }
"""),
    ]


def css_blocks(p: dict) -> List[str]:
    """Every stylesheet for the active palette, dark overrides first.

    Dark rules are emitted before the custom rules so the custom classes win
    any specificity tie, and each block stays under :data:`CSS_BLOCK_LIMIT` so
    Streamlit cannot silently truncate it.
    """
    blocks = _dark_blocks(p) + _custom_blocks(p)
    oversized = [len(b) for b in blocks if len(b) > CSS_BLOCK_LIMIT]
    if oversized:  # pragma: no cover - guard against a partial render
        raise RuntimeError(f"CSS block exceeds safe size: {oversized}")
    return blocks


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def fmt_int(value: Optional[float]) -> str:
    """Thousands-separated integer, tolerant of NaN / None."""
    return "—" if _is_missing(value) else f"{float(value):,.0f}"


def fmt_date(value: Any) -> str:
    return "—" if _is_missing(value) else f"{pd.Timestamp(value):%d %b %Y}"


def fmt_datetime(value: Any) -> str:
    return "—" if _is_missing(value) else f"{pd.Timestamp(value):%d %b %Y, %H:%M}"


def fmt_compact(value: Optional[float]) -> str:
    """1.20M / 340.1k / 812 - keeps long counts readable inside a sentence."""
    if _is_missing(value):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}k"
    return f"{value:,.0f}"


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------


def section(title: str, note: str = "") -> None:
    """Render a numbered section header with an optional explanatory line."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def info_strip(html: str, kind: str = "info") -> None:
    css = "info-strip" if kind == "info" else "warn-strip"
    st.markdown(f'<div class="{css}">{html}</div>', unsafe_allow_html=True)


def caption(text: str) -> None:
    st.markdown(f'<div class="caption-note">{text}</div>', unsafe_allow_html=True)


def base_layout(
    title: str,
    x_title: str,
    y_title: str,
    height: int = 420,
    hovermode: str = "x unified",
) -> go.Layout:
    """Shared Plotly layout so every chart in the app looks like one system."""
    return go.Layout(
        title=dict(text=title, font=dict(size=16, color=P["text"]), x=0.01, xanchor="left"),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12, color=P["body"])),
            gridcolor=P["grid"],
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikedash="dot",
            linecolor=P["grid"],
            tickfont=dict(color=P["body"]),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12, color=P["body"])),
            gridcolor=P["grid"],
            zeroline=False,
            rangemode="tozero",
            linecolor=P["grid"],
            tickfont=dict(color=P["body"]),
        ),
        hovermode=hovermode,
        height=height,
        margin=dict(l=10, r=10, t=52, b=10),
        plot_bgcolor=P["bg"],
        paper_bgcolor=P["paper"],
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=12, color=P["body"]),
        dragmode="zoom",
        showlegend=False,
        hoverlabel=dict(
            bgcolor=P["hover_bg"],
            bordercolor=P["hover_border"],
            font=dict(color=P["text"], size=12),
        ),
    )


def line_figure(
    series: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_title: str,
    y_title: str,
    color: Optional[str] = None,
    mode: str = "lines",
    height: int = 420,
    hover_fmt: str = "%{x|%d %b %Y, %H:%M}<br>Ridership: %{y:,.0f}",
) -> go.Figure:
    """A single time-series line, downsampled to keep the figure small.

    ``connectgaps=False`` keeps absent timestamps visible as breaks in the line
    instead of drawing an invented straight line across them.
    """
    plotted = pp.downsample_for_plot(
        series[[x_col, y_col]].dropna(), HOURLY_POINT_BUDGET
    )
    stroke = color or P["primary"]
    fig = go.Figure(
        go.Scatter(
            x=plotted[x_col],
            y=plotted[y_col],
            mode=mode,
            line=dict(color=stroke, width=1.7),
            marker=dict(size=4, color=stroke) if "markers" in mode else None,
            connectgaps=False,
            hovertemplate=f"<b>{hover_fmt}</b><extra></extra>",
        )
    )
    fig.update_layout(**base_layout(title, x_title, y_title, height=height).to_plotly_json())
    return fig


def hbar_figure(
    frame: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    top_n: int = 10,
    color: Optional[str] = None,
    hover_suffix: str = "riders",
) -> go.Figure:
    """Horizontal bar chart - long station names read correctly this way."""
    if frame is None or frame.empty:
        return go.Figure()
    top = frame.nlargest(top_n, value_col).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=top[value_col],
            y=top[label_col],
            orientation="h",
            marker=dict(color=color or P["primary"], line=dict(width=0)),
            hovertemplate=f"<b>%{{y}}</b><br>%{{x:,.0f}} {hover_suffix}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=P["text"]), x=0.01, xanchor="left"),
        xaxis=dict(title=None, gridcolor=P["grid"], zeroline=False,
                   linecolor=P["grid"], tickfont=dict(color=P["body"])),
        yaxis=dict(title=None, gridcolor="rgba(0,0,0,0)", automargin=True,
                   tickfont=dict(color=P["body"])),
        height=max(320, 30 * len(top) + 130),
        margin=dict(l=10, r=20, t=50, b=10),
        plot_bgcolor=P["bg"],
        paper_bgcolor=P["paper"],
        showlegend=False,
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=12, color=P["body"]),
        hoverlabel=dict(
            bgcolor=P["hover_bg"], bordercolor=P["hover_border"],
            font=dict(color=P["text"], size=12),
        ),
    )
    return fig


def vbar_figure(
    frame: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    x_title: str,
    y_title: str,
    colors: Optional[Sequence[str]] = None,
    height: int = 380,
    hover_fmt: str = "<b>%{x}</b><br>Average: %{y:,.0f}",
) -> go.Figure:
    """Vertical bar chart used for the categorical profiles."""
    if frame is None or frame.empty:
        return go.Figure()
    fig = go.Figure(
        go.Bar(
            x=frame[label_col].astype(str),
            y=frame[value_col],
            marker=dict(color=list(colors) if colors is not None else P["primary"]),
            hovertemplate=f"{hover_fmt}<extra></extra>",
        )
    )
    fig.update_layout(**base_layout(
        title, x_title, y_title, height=height, hovermode="x"
    ).to_plotly_json())
    return fig


def station_map_figure(points: pd.DataFrame, height: int = 640) -> go.Figure:
    """Interactive station map: one point per complex, sized by total ridership.

    Uses ``Scattergeo``, which renders the built-in geographic projection and
    therefore needs no Mapbox access token. Positions come from the dataset's
    own ``latitude``/``longitude``; nothing is geocoded externally.
    """
    if points is None or points.empty:
        fig = go.Figure()
        fig.update_layout(
            **base_layout(
                "MTA Station Ridership Map", "Longitude", "Latitude", height=height
            ).to_plotly_json()
        )
        return fig

    # Bubble *area* is proportional to volume, so the visual encoding is
    # honest: sqrt of the value scaled to a 8..50px diameter range.
    volumes = pd.to_numeric(points["Total_Ridership"], errors="coerce").fillna(0).clip(lower=0)
    scale = float(volumes.max()) or 1.0
    sizes = 8 + 42 * np.sqrt(volumes / scale)

    boroughs = sorted(points["borough"].dropna().unique())
    hover = (
        "<b>%{customdata[0]}</b><br>"
        "Borough: %{customdata[1]}<br>"
        "Total ridership: %{customdata[2]:,.0f}<br>"
        "Average hourly: %{customdata[3]:,.0f}<br>"
        "Latitude: %{lat:.5f}<br>"
        "Longitude: %{lon:.5f}"
        "<extra></extra>"
    )
    custom = np.stack(
        [
            points["station_complex"].astype(str).to_numpy(),
            points["borough"].astype(str).to_numpy(),
            pd.to_numeric(points["Total_Ridership"], errors="coerce").fillna(0).to_numpy(),
            pd.to_numeric(points["Avg_Hourly_Ridership"], errors="coerce").fillna(0).to_numpy(),
        ],
        axis=-1,
    )

    fig = go.Figure()
    for borough in boroughs:
        mask = (points["borough"] == borough).to_numpy()
        fig.add_trace(
            go.Scattergeo(
                lat=points.loc[mask, "latitude"],
                lon=points.loc[mask, "longitude"],
                mode="markers",
                name=borough,
                marker=dict(
                    size=sizes.to_numpy()[mask],
                    color=BOROUGH_COLORS.get(borough, BOROUGH_FALLBACK),
                    opacity=0.85,
                    line=dict(width=0.6, color=P["bg"]),
                ),
                customdata=custom[mask],
                hovertemplate=hover,
            )
        )

    fig.update_layout(
        title=dict(
            text="MTA Station Ridership Map",
            font=dict(size=16, color=P["text"]),
            x=0.01,
            xanchor="left",
        ),
        height=height,
        margin=dict(l=0, r=0, t=52, b=0),
        paper_bgcolor=P["paper"],
        plot_bgcolor=P["bg"],
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=12, color=P["body"]),
        showlegend=True,
        legend=dict(
            title=dict(text="Borough", font=dict(size=12, color=P["body"])),
            font=dict(size=11, color=P["body"]),
            bgcolor=P["hover_bg"],
        ),
        hoverlabel=dict(
            bgcolor=P["hover_bg"], bordercolor=P["hover_border"],
            font=dict(color=P["text"], size=12),
        ),
        geo=dict(
            # Framed on the city, which is where every station actually is.
            scope="north america",
            projection_type="albers usa",
            showland=True,
            landcolor=P["panel"],
            showlakes=False,
            showocean=False,
            showframe=False,
            showcoastlines=False,
            lataxis=dict(range=[40.45, 41.05]),
            lonaxis=dict(range=[-74.30, -73.65]),
            bgcolor=P["bg"],
        ),
    )
    return fig


# --------------------------------------------------------------------------
# Cached data access
#
# Two cache layers sit behind these helpers:
#   1. `@st.cache_data` - avoids re-reading Parquet on every rerun.
#   2. `data_loader.CACHE` - the Parquet files themselves, which also skip the
#      API call and survive a restart of the app.
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def get_domains() -> Dict[str, List[str]]:
    return dl.load_domains()


@st.cache_data(show_spinner=False)
def get_station_meta() -> pd.DataFrame:
    return dl.load_station_meta()


@st.cache_data(show_spinner=False)
def get_raw_row_count() -> int:
    return dl.load_raw_row_count()


@st.cache_data(show_spinner=False)
def get_hourly(filters: Filters) -> pd.DataFrame:
    return dl.load_hourly(filters)


@st.cache_data(show_spinner=False)
def get_station_totals(filters: Filters) -> pd.DataFrame:
    return dl.load_station_totals(filters)


@st.cache_data(show_spinner=False)
def get_borough_totals(filters: Filters) -> pd.DataFrame:
    return dl.load_borough_totals(filters)


@st.cache_data(show_spinner=False)
def get_payment_totals(filters: Filters) -> pd.DataFrame:
    return dl.load_payment_totals(filters)


@st.cache_data(show_spinner=False)
def get_fare_totals(filters: Filters) -> pd.DataFrame:
    return dl.load_fare_totals(filters)


@st.cache_data(show_spinner=False)
def get_mode_totals(filters: Filters) -> pd.DataFrame:
    return dl.load_mode_totals(filters)


@st.cache_data(show_spinner=False)
def get_station_hourly(filters: Filters, station_id: str) -> pd.DataFrame:
    return dl.load_station_hourly(filters, station_id)


@st.cache_data(show_spinner=False)
def get_cache_inventory() -> pd.DataFrame:
    return dl.cache_status()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

ALL = "All"

#: Hover format per trend grain, so a monthly chart shows "Jan 2024" and an
#: hourly one shows "29 Oct 2024 17:00" rather than both showing a date only.
TREND_HOVER: Dict[str, str] = {
    "Hourly": "%{x|%d %b %Y, %H:%M}<br>Ridership: %{y:,.0f}",
    "Daily": "%{x|%d %b %Y}<br>Ridership: %{y:,.0f}",
    "Weekly": "%{x|%d %b %Y}<br>Ridership: %{y:,.0f}",
    "Monthly": "%{x|%b %Y}<br>Ridership: %{y:,.0f}",
}


def _default_dates() -> Tuple[date, date]:
    lo, hi = dl.dataset_span()
    start = max(lo.normalize(), dl.DEFAULT_START)
    end = min(hi.normalize(), dl.DEFAULT_END)
    if start > end:  # defensive: fall back to the full span
        start, end = lo.normalize(), hi.normalize()
    return start.date(), end.date()


def _reset_filters() -> None:
    """Restore every sidebar widget to its default.

    Registered as an ``on_click`` callback: it runs *before* the next script
    execution, so the widgets see carried-over state rather than a conflicting
    value-plus-default combination.
    """
    start, end = _default_dates()
    st.session_state.update(
        {
            "sb_dates": (start, end),
            "sb_borough": [],
            "sb_station": [],
            "sb_mode": [],
            "sb_payment": [],
            "sb_fare": [],
            "sb_agg": "Auto",
            "sb_top_n": 10,
        }
    )


def build_sidebar(meta: pd.DataFrame, domains: Dict[str, List[str]]) -> Dict[str, Any]:
    """Render every filter and return them in the shape `Filters` expects."""
    lo, hi = dl.dataset_span()
    lo_date, hi_date = lo.normalize().date(), hi.normalize().date()
    start, end = _default_dates()

    names: Dict[str, str] = {}
    boroughs_by_id: Dict[str, str] = {}
    if meta is not None and not meta.empty:
        names = dict(
            zip(meta["station_complex_id"].astype(str), meta["station_complex"].astype(str))
        )
        boroughs_by_id = dict(
            zip(meta["station_complex_id"].astype(str), meta["borough"].astype(str))
        )
    station_ids = sorted(names, key=lambda i: (names[i].lower(), i))

    def station_label(station_id: str) -> str:
        name = names.get(station_id, f"Complex {station_id}")
        borough = boroughs_by_id.get(station_id)
        return f"{name} ({borough})" if borough else name

    with st.sidebar:
        st.markdown("### Appearance")
        st.toggle(
            "Dark mode",
            key="dark_mode",
            help="Switch between the light and dark presentation themes.",
        )
        st.caption(f"Currently showing the **{current_theme()}** theme.")
        st.divider()

        st.markdown("### Filters")
        st.caption("Every figure below updates to match this selection.")

        dates = st.date_input(
            "Date range",
            value=(start, end),
            min_value=lo_date,
            max_value=hi_date,
            format="DD/MM/YYYY",
            key="sb_dates",
            help=f"The dataset spans {fmt_date(lo)} to {fmt_date(hi)}.",
        )
        if isinstance(dates, (date, datetime)):
            dates = (dates, dates)
        elif isinstance(dates, (list, tuple)) and len(dates) == 1:
            dates = (dates[0], dates[0])

        boroughs = st.multiselect(
            "Borough",
            options=list(domains.get("borough", [])),
            key="sb_borough",
            help=f"{ALL} boroughs (leave empty).",
        )
        stations = st.multiselect(
            "Station / Station Complex",
            options=station_ids,
            format_func=station_label,
            key="sb_station",
            help=f"{ALL} complexes (leave empty). The box is searchable by name.",
        )
        modes = st.multiselect(
            "Transit mode", options=list(domains.get("transit_mode", [])), key="sb_mode"
        )
        payments = st.multiselect(
            "Payment method", options=list(domains.get("payment_method", [])), key="sb_payment"
        )
        fares = st.multiselect(
            "Fare class category",
            options=list(domains.get("fare_class_category", [])),
            key="sb_fare",
        )
        aggregation = st.radio(
            "Time aggregation",
            options=["Auto", *pp.AGGREGATION_ORDER],
            key="sb_agg",
            help=(
                "Auto uses Hourly for short windows and Daily or Monthly for long "
                "ones, so a chart never carries an unreadable number of points."
            ),
        )

        st.divider()
        st.button("Reset filters", width="stretch", on_click=_reset_filters)

    return {
        "dates": dates,
        "boroughs": list(boroughs),
        "stations": list(stations),
        "modes": list(modes),
        "payments": list(payments),
        "fares": list(fares),
        "aggregation": aggregation,
    }


def make_filters(sidebar: Dict[str, Any]) -> Filters:
    """Translate the sidebar state into a :class:`Filters`.

    An empty multiselect means "no restriction" (``None``), which is the usual
    Streamlit idiom and matches the documented "default: All".
    """
    dates = sidebar["dates"]
    return Filters.build(
        start=dates[0],
        end=dates[-1],
        boroughs=sidebar["boroughs"] or None,
        stations=sidebar["stations"] or None,
        modes=sidebar["modes"] or None,
        payments=sidebar["payments"] or None,
        fares=sidebar["fares"] or None,
    )


def resolve_aggregation(requested: str, days: int) -> str:
    """Pick the chart grain for the main trend from the *length of the window*.

    Density is what makes a long-range chart unreadable: a daily line over
    three years is ~1,100 near-identical points. The thresholds are therefore
    expressed in calendar days:

    * more than 90 days  -> Monthly (month -> SUM(ridership))
    * 15 to 90 days      -> Daily (date -> SUM(ridership))
    * 14 days or fewer   -> Hourly (transit_timestamp -> SUM(ridership))

    An explicit sidebar choice always wins, so the user can still inspect a
    finer grain over a long window if they want to.
    """
    if requested != "Auto":
        return requested
    if days > 90:
        return "Monthly"
    if days > 14:
        return "Daily"
    return "Hourly"


# --------------------------------------------------------------------------
# Shared view model
#
# Every tab reads from one :class:`View`, so each cube is fetched and each
# derived series computed exactly once per rerun.
# --------------------------------------------------------------------------


@dataclass
class View:
    filters: Filters
    hourly: pd.DataFrame
    station_totals: pd.DataFrame
    borough_totals: pd.DataFrame
    payment_totals: pd.DataFrame
    fare_totals: pd.DataFrame
    mode_totals: pd.DataFrame
    meta: pd.DataFrame

    kpis: Dict[str, Any]
    hour_prof: pd.DataFrame
    dow_prof: pd.DataFrame
    matrix: pd.DataFrame
    years: pd.DataFrame
    months: pd.DataFrame
    stations: pd.DataFrame
    boroughs: pd.DataFrame
    payments: pd.DataFrame
    fares: pd.DataFrame
    aggregation: str
    raw_rows: int


def _window_days(filters: Filters) -> int:
    """Number of calendar days in the selected window (inclusive)."""
    return int((filters.hi - filters.lo).days) + 1


def _period_label(filters: Filters) -> str:
    """Human-readable description of the active selection, for the observations.

    Names the date window *and* any dimension filters, so a sentence such as
    "Within the selected period..." is never ambiguous about what it covers.
    """
    label = f"the selected period ({filters.lo:%d %b %Y} to {filters.hi:%d %b %Y}"
    bits = []
    for name, values in (
        ("borough", filters.boroughs),
        ("station", filters.stations),
        ("mode", filters.modes),
        ("payment method", filters.payments),
        ("fare class", filters.fares),
    ):
        if values:
            bits.append(f"{name}={'+'.join(values) if len(values) <= 2 else f'{len(values)} {name}s'}")
    if bits:
        label += ", " + ", ".join(bits)
    return label + ")"


def _rename_totals(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[key, "Total_Ridership", "Total_Transfers", "Raw_Rows"])
    return frame.rename(
        columns={
            "ridership": "Total_Ridership",
            "transfers": "Total_Transfers",
            "raw_rows": "Raw_Rows",
        }
    )


def build_view(sidebar: Dict[str, Any]) -> View:
    """Fetch every cube for the current filter set and derive the view."""
    filters = make_filters(sidebar)

    hourly = get_hourly(filters)
    station_totals = get_station_totals(filters)
    borough_totals = get_borough_totals(filters)
    payment_totals = get_payment_totals(filters)
    fare_totals = get_fare_totals(filters)
    mode_totals = get_mode_totals(filters)
    meta = get_station_meta()

    # How many raw API rows sit behind this selection? Used only to describe
    # the selection honestly in the data-quality tab.
    raw_rows = 0
    for frame in (station_totals, borough_totals, payment_totals, fare_totals, mode_totals):
        if frame is not None and not frame.empty and "raw_rows" in frame.columns:
            value = pd.to_numeric(frame["raw_rows"], errors="coerce").fillna(0).max()
            raw_rows = max(raw_rows, int(value))

    stations = an.station_summary(station_totals, meta, window_hours=int(len(hourly)))

    return View(
        filters=filters,
        hourly=hourly,
        station_totals=station_totals,
        borough_totals=borough_totals,
        payment_totals=payment_totals,
        fare_totals=fare_totals,
        mode_totals=mode_totals,
        meta=meta,
        kpis=an.compute_kpis(hourly, station_totals, borough_totals),
        hour_prof=an.hourly_profile(hourly),
        dow_prof=an.dow_profile(hourly),
        matrix=an.hour_day_matrix(hourly),
        years=an.year_trend(hourly),
        months=an.monthly_trend(hourly),
        stations=stations,
        # The borough station counts come from the *joined* station summary,
        # which is the only frame carrying a borough per complex.
        boroughs=an.borough_summary(borough_totals, stations),
        payments=_rename_totals(payment_totals, "payment_method"),
        fares=_rename_totals(fare_totals, "fare_class_category"),
        aggregation=resolve_aggregation(
            sidebar["aggregation"], _window_days(filters)
        ),
        raw_rows=raw_rows,
    )


# --------------------------------------------------------------------------
# Tab 1 - Overview
# --------------------------------------------------------------------------


def render_overview(v: View) -> None:
    kpis = v.kpis
    section(
        "1 &middot; Overall Ridership Overview",
        "Every KPI is recomputed from the current filter selection. Ridership is "
        "summed server-side across every payment method and fare class, so a raw "
        "row is never counted as a separate observation.",
    )

    if v.hourly.empty:
        st.warning(
            "No observations match the current filter selection. Widen the date "
            "range or clear a filter in the sidebar.",
            icon="\U0001F6A8",
        )
        return

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Ridership", fmt_compact(kpis["total_ridership"]),
              help=f"{fmt_int(kpis['total_ridership'])} riders summed over the selection.")
    r2.metric("Average Hourly Ridership", fmt_int(kpis["avg_hourly"]),
              help="Mean of the citywide total at each hourly timestamp.")
    r3.metric("Peak Hourly Ridership", fmt_compact(kpis["max_hourly"]),
              help="Largest single hourly citywide total.")
    r4.metric("Number of Stations", fmt_int(kpis["n_stations"]),
              help="Distinct station complexes reporting in the selection.")

    r5, r6, r7, r8 = st.columns(4)
    r5.metric("Number of Boroughs", fmt_int(kpis["n_boroughs"]))
    r6.metric("Number of Observations", fmt_int(kpis["n_observations"]),
              help="Distinct hourly timestamps analysed - the analysis records, not raw rows.")
    r7.metric(
        "Selected Date Range",
        f"{pd.Timestamp(kpis['min_timestamp']):%d %b %Y} → "
        f"{pd.Timestamp(kpis['max_timestamp']):%d %b %Y}",
        help="First to last hourly timestamp present in the selection.",
    )
    r8.metric("Total Transfers", fmt_compact(kpis["total_transfers"]),
              help=f"{fmt_int(kpis['total_transfers'])} recorded transfers.")

    r9, r10 = st.columns(2)
    r9.metric("Average Transfers", fmt_int(kpis["avg_transfers"]),
              help="Mean transfers per hourly timestamp.")
    r10.metric("Peak Timestamp", fmt_datetime(kpis["max_hourly_timestamp"]),
               help="When the peak hourly total occurred.")

    caption(
        "Observations counts <b>hourly timestamps</b>. The underlying raw dataset "
        f"contains {fmt_compact(get_raw_row_count())} rows, split across payment method "
        "and fare class; those rows are aggregated on the server and never loaded."
    )

    st.divider()
    section(
        "2 &middot; Overall Time Series",
        f"Shown at <b>{v.aggregation.lower()}</b> resolution, auto-selected for a "
        f"{_window_days(v.filters)}-day window. Zoom, pan or box-select to explore.",
    )

    days = _window_days(v.filters)
    series = pp.resample_series(v.hourly, v.aggregation)
    if series.empty:
        st.info("No data available at this aggregation level.")
    else:
        st.plotly_chart(
            line_figure(
                series, "period_start", "ridership",
                f"MTA {v.aggregation} Ridership Trend",
                "Period start", "Total Ridership (all selected stations)",
                height=470,
                hover_fmt=TREND_HOVER[v.aggregation],
            ),
            width="stretch", config=PLOTLY_CONFIG,
        )

        # The finer grains stay reachable: a long window hides the hourly
        # detail (26k points would be unreadable) but never removes it.
        if days > 90:
            caption(
                "The window spans more than 90 days, so the trend above is aggregated "
                "monthly. Use the Time aggregation selector in the sidebar to force a "
                "daily or hourly view."
            )

        with st.expander(
            "Inspect a finer time scale (daily / monthly / hourly)", expanded=False
        ):
            st.markdown("**Daily Ridership Trend**")
            daily = pp.resample_series(v.hourly, "Daily")
            st.plotly_chart(
                line_figure(daily, "period_start", "ridership", "Daily Ridership Trend",
                            "Date", "Total Daily Ridership", color=P["accent"],
                            mode="lines+markers", height=360,
                            hover_fmt="%{x|%d %b %Y}<br>Total: %{y:,.0f}"),
                width="stretch", config=PLOTLY_CONFIG,
            )

            st.markdown("**Monthly Ridership Trend**")
            monthly = pp.resample_series(v.hourly, "Monthly")
            fig_month = go.Figure(
                go.Bar(
                    x=monthly["period_start"],
                    y=monthly["ridership"],
                    marker=dict(color=P["primary"]),
                    hovertemplate="<b>%{x|%b %Y}</b><br>Total: %{y:,.0f}<extra></extra>",
                )
            )
            fig_month.update_layout(**base_layout(
                "Monthly Ridership Trend", "Month", "Total Ridership",
                height=360, hovermode="x",
            ).to_plotly_json())
            st.plotly_chart(fig_month, width="stretch", config=PLOTLY_CONFIG)

            st.markdown("**Hourly Ridership Trend**")
            hourly = pp.resample_series(v.hourly, "Hourly")
            plotted = pp.downsample_for_plot(hourly, HOURLY_POINT_BUDGET)
            st.plotly_chart(
                line_figure(plotted, "period_start", "ridership", "Hourly Ridership Trend",
                            "Timestamp", "Total hourly ridership", height=360),
                width="stretch", config=PLOTLY_CONFIG,
            )
            caption(
                f"{len(plotted):,} of {len(hourly):,} hourly points plotted "
                "(every nth point is kept so the shape is preserved)."
            )

    st.divider()
    section(
        "Initial Observations",
        "Generated automatically from the values computed for the current "
        "selection. Descriptive only — no causal claims are made.",
    )
    for note in an.build_observations(
        v.kpis, v.hour_prof, v.dow_prof, v.stations, v.boroughs,
        v.years, v.payments, None, v.months, _period_label(v.filters),
    ):
        st.markdown(f"- {note}")


# --------------------------------------------------------------------------
# Tab 2 - Temporal analysis
# --------------------------------------------------------------------------


def render_temporal(v: View) -> None:
    section(
        "3 &middot; Temporal Patterns",
        "Averages are computed over the citywide hourly series: at each timestamp "
        "every station is summed first, then the hourly totals are averaged by "
        "hour, weekday or month.",
    )
    if v.hourly.empty:
        st.info("No observations match the current filter selection.")
        return

    # --- A -------------------------------------------------------------
    st.markdown("**A &middot; Average Ridership by Hour of Day**")
    hours = v.hour_prof.copy()
    hours["Label"] = hours["hour"].apply(lambda h: f"{int(h):02d}:00")
    st.plotly_chart(
        vbar_figure(hours, "Label", "Avg_Ridership",
                    "Average Ridership by Hour of Day", "Hour of Day",
                    "Average hourly ridership", height=380),
        width="stretch", config=PLOTLY_CONFIG,
    )
    caption(
        "Each bar is the mean of the citywide hourly totals falling in that hour, so "
        "it answers “how busy is the network at 08:00 on a typical day?”."
    )

    # --- B -------------------------------------------------------------
    st.markdown("**B &middot; Average Ridership by Day of Week**")
    dow = v.dow_prof.copy()
    dow["DayName"] = dow["DayName"].astype(object)
    dow = dow.sort_values("DayOfWeek")
    st.plotly_chart(
        vbar_figure(
            dow, "DayName", "Avg_Ridership",
            "Average Ridership by Day of Week", "Day of Week",
            "Average hourly ridership",
            colors=[P["accent"] if d in ("Saturday", "Sunday") else P["primary"]
                    for d in dow["DayName"]],
            height=380,
        ),
        width="stretch", config=PLOTLY_CONFIG,
    )
    caption("Monday-first ordering as required; weekend days are highlighted.")

    # --- C -------------------------------------------------------------
    st.markdown("**C &middot; Ridership Heatmap — Day vs Hour**")
    matrix = v.matrix.copy()
    matrix.index = [DOW_NUMBER_TO_NAME.get(i, "?") for i in matrix.index]
    heat = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=[f"{int(c):02d}:00" for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            colorscale=P["heat_scale"],
            # NB: the Plotly property is spelled `reversescale`; a misspelling
            # is silently discarded rather than raising.
            reversescale=P["heat_reverse"],
            hovertemplate="<b>%{y}</b> at <b>%{x}</b><br>Average: %{z:,.0f}<extra></extra>",
            colorbar=dict(
                title=dict(text="Avg<br>ridership", font=dict(size=11, color=P["body"])),
                thickness=14, len=0.85,
                tickfont=dict(color=P["body"], size=11), outlinecolor=P["grid"],
            ),
        )
    )
    heat.update_layout(**base_layout(
        "Ridership Heatmap — Day vs Hour", "Hour of Day", "Day of Week", height=440
    ).to_plotly_json())
    heat.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    heat.update_yaxes(autorange="reversed")
    st.plotly_chart(heat, width="stretch", config=PLOTLY_CONFIG)

    # --- D -------------------------------------------------------------
    st.divider()
    st.markdown("**D &middot; Monthly and Yearly Trend**")
    years = v.years
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("**Ridership by Year**")
        if years.empty:
            st.info("No yearly breakdown available.")
        else:
            fig_year = go.Figure(
                go.Bar(
                    x=years["year"].astype(str),
                    y=years["Total_Ridership"],
                    marker=dict(color=P["primary"]),
                    customdata=np.stack(
                        [years["Avg_Hourly_Ridership"], years["Observations"]], axis=-1
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>Total: %{y:,.0f}<br>"
                        "Average hourly: %{customdata[0]:,.0f}<br>"
                        "Observations: %{customdata[1]:,.0f}<extra></extra>"
                    ),
                )
            )
            fig_year.update_layout(**base_layout(
                "Total Ridership by Calendar Year", "Year", "Total Ridership",
                height=380, hovermode="x",
            ).to_plotly_json())
            fig_year.update_xaxes(type="category")
            st.plotly_chart(fig_year, width="stretch", config=PLOTLY_CONFIG)
            if len(years) >= 2:
                first, last = years.iloc[0], years.iloc[-1]
                pct = (
                    100.0 * (last["Total_Ridership"] - first["Total_Ridership"])
                    / first["Total_Ridership"]
                    if first["Total_Ridership"] else float("nan")
                )
                info_strip(
                    f"The {int(last['year'])} total is <b>{abs(pct):.1f}% "
                    f"{'higher' if pct >= 0 else 'lower'}</b> than {int(first['year'])}. "
                    "Partial years are shown as recorded and are <b>not</b> scaled."
                )

    with c2:
        st.markdown("**Average Hourly Ridership by Year**")
        if not years.empty:
            fig_avg = go.Figure(
                go.Scatter(
                    x=years["year"].astype(str),
                    y=years["Avg_Hourly_Ridership"],
                    mode="lines+markers",
                    line=dict(color=P["accent"], width=2),
                    marker=dict(size=7, color=P["accent"]),
                    connectgaps=False,
                    hovertemplate="<b>%{x}</b><br>Average hourly: %{y:,.0f}<extra></extra>",
                )
            )
            fig_avg.update_layout(**base_layout(
                "Average Hourly Ridership by Year", "Year",
                "Average hourly ridership", height=380,
            ).to_plotly_json())
            fig_avg.update_xaxes(type="category")
            st.plotly_chart(fig_avg, width="stretch", config=PLOTLY_CONFIG)

    st.markdown("**Ridership by Month (all selected years)**")
    if not v.months.empty:
        fig_m = go.Figure(
            go.Bar(
                x=v.months["YearMonth"].astype(str),
                y=v.months["Total_Ridership"],
                marker=dict(color=P["primary"]),
                hovertemplate="<b>%{x}</b><br>Total: %{y:,.0f}<extra></extra>",
            )
        )
        fig_m.update_layout(**base_layout(
            "Total Ridership by Calendar Month", "Month", "Total Ridership",
            height=340, hovermode="x",
        ).to_plotly_json())
        # Period strings like "2024-07" are auto-parsed by Plotly into a
        # continuous *date* axis; force a categorical axis so only the months
        # that actually occur are shown.
        fig_m.update_xaxes(type="category")
        st.plotly_chart(fig_m, width="stretch", config=PLOTLY_CONFIG)


# --------------------------------------------------------------------------
# Tab 3 - Station analysis
# --------------------------------------------------------------------------


def render_station(v: View) -> None:
    if v.stations.empty:
        st.info("No station complexes match the current filter selection.")
        return

    options_borough = [ALL] + sorted(
        str(b) for b in v.stations["borough"].dropna().unique() if b != "Unknown"
    )
    stale = st.session_state.get("sb_borough_focus")
    if stale is not None and stale not in options_borough:
        del st.session_state["sb_borough_focus"]

    borough_focus = st.selectbox(
        "Borough focus",
        options=options_borough,
        key="sb_borough_focus",
        help="Narrows the station ranking and the drill-down below to one borough.",
    )
    scoped = (
        v.stations if borough_focus == ALL
        else v.stations.loc[v.stations["borough"] == borough_focus]
    )
    if scoped.empty:
        st.info(f"No station complexes found in {borough_focus} for this selection.")
        return

    section(
        "4 &middot; Top Stations by Total Ridership",
        f"Ranked across the current selection"
        f"{'' if borough_focus == ALL else f' ({borough_focus} only)'}.",
    )
    top_n = st.slider(
        "Number of stations to display",
        min_value=5, max_value=20, value=10, step=5, key="sb_top_n",
        help="Applies to both ranking charts.",
    )
    left, right = st.columns(2, gap="medium")
    with left:
        st.plotly_chart(
            hbar_figure(scoped, "station_complex", "Total_Ridership",
                        "Top Station Complexes by Total Ridership", top_n),
            width="stretch", config=PLOTLY_CONFIG,
        )
    with right:
        st.plotly_chart(
            hbar_figure(scoped, "station_complex", "Total_Transfers",
                        f"Top {top_n} Station Complexes by Recorded Transfers", top_n,
                        color=P["accent"], hover_suffix="transfers"),
            width="stretch", config=PLOTLY_CONFIG,
        )

    ranking = scoped.nlargest(top_n, "Total_Ridership").reset_index(drop=True)
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    st.markdown(f"**Station Ranking — Top {top_n} by Total Ridership**")
    st.dataframe(
        ranking[["Rank", "station_complex", "borough", "Total_Ridership",
                 "Avg_Hourly_Ridership"]],
        hide_index=True, width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "station_complex": st.column_config.TextColumn("Station Complex", width="large"),
            "borough": st.column_config.TextColumn("Borough"),
            "Total_Ridership": st.column_config.NumberColumn("Total Ridership", format="%,d"),
            "Avg_Hourly_Ridership": st.column_config.NumberColumn(
                "Average Hourly Ridership", format="%,.0f",
                help="Total ridership divided by the number of hourly timestamps in the "
                     "selected window."),
        },
    )
    with st.expander("Transfers and underlying row counts", expanded=False):
        st.dataframe(
            ranking[["Rank", "station_complex", "borough", "Total_Transfers", "Raw_Rows"]],
            hide_index=True, width="stretch",
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "station_complex": st.column_config.TextColumn("Station Complex", width="large"),
                "borough": st.column_config.TextColumn("Borough"),
                "Total_Transfers": st.column_config.NumberColumn("Transfers", format="%,d"),
                "Raw_Rows": st.column_config.NumberColumn(
                    "Raw API rows",
                    help="Underlying dataset rows summed into this station's total. This is "
                         "roughly ten times the number of observed hours, because a "
                         "station-hour is split across payment methods and fare classes."),
            },
        )
    caption(
        "Each total is a server-side sum over the station's raw rows, so the "
        "payment-method and fare-class split is already collapsed."
    )

    st.divider()
    scoped_ids = [str(i) for i in scoped["station_complex_id"]]
    label_by_id = dict(zip(scoped["station_complex_id"], scoped["station_complex"]))
    focus_id = st.selectbox(
        "Select a station / station complex",
        options=scoped_ids,
        format_func=lambda i: label_by_id.get(i, f"Complex {i}"),
        key="sb_focus",
        help="Loads the hourly series for the chosen complex.",
    )
    if not focus_id:
        return
    row = scoped.loc[scoped["station_complex_id"] == focus_id]
    if row.empty:
        return
    row = row.iloc[0]
    focus_name = str(row["station_complex"])

    series = get_station_hourly(v.filters, focus_id)
    if series.empty:
        st.warning("No hourly observations for this station in the current window.")
        return

    section(f"Station Drill-Down — {focus_name}",
            f"{row['borough']} · station complex {focus_id}")

    prepared = pp.add_calendar_columns(series, "timestamp")
    hour_means = prepared.groupby("hour", observed=True)["ridership"].mean()
    peak_row = series.loc[series["ridership"].idxmax()]

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Total Ridership", fmt_compact(series["ridership"].sum()))
    d2.metric("Average Hourly Ridership", fmt_int(series["ridership"].mean()))
    d3.metric("Peak Hour", f"{int(hour_means.idxmax()):02d}:00",
              help="Hour of day with the highest average ridership.")
    d4.metric("Peak Ridership", fmt_compact(peak_row["ridership"]),
              help="Largest single hourly reading.")
    d5.metric("Total Transfers", fmt_compact(series["transfers"].sum()))

    st.markdown("**1 &middot; Hourly Trend**")
    st.plotly_chart(
        line_figure(series, "timestamp", "ridership", f"Hourly Ridership — {focus_name}",
                    "Timestamp", "Ridership", height=380),
        width="stretch", config=PLOTLY_CONFIG,
    )

    st.markdown("**2 &middot; Daily Trend**")
    st.plotly_chart(
        line_figure(pp.resample_series(series, "Daily"), "period_start", "ridership",
                    f"Daily Ridership — {focus_name}", "Date", "Daily ridership",
                    color=P["accent"], mode="lines+markers", height=330,
                    hover_fmt="%{x|%d %b %Y}<br>Total: %{y:,.0f}"),
        width="stretch", config=PLOTLY_CONFIG,
    )

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        st.markdown("**3 &middot; Average Ridership by Hour**")
        sh = prepared.groupby("hour", observed=True)["ridership"].mean().reindex(range(24))
        sh = sh.reset_index()
        sh["Label"] = sh["hour"].apply(lambda h: f"{int(h):02d}:00")
        st.plotly_chart(
            vbar_figure(sh, "Label", "ridership",
                        f"Average Ridership by Hour — {focus_name}", "Hour of Day",
                        "Average ridership", height=340),
            width="stretch", config=PLOTLY_CONFIG,
        )
    with c4:
        st.markdown("**4 &middot; Day-of-Week Pattern**")
        sd = prepared.groupby("day_of_week", observed=True)["ridership"].mean().reindex(range(7))
        sd = sd.reset_index()
        sd["DayName"] = sd["day_of_week"].map(DOW_NUMBER_TO_NAME)
        st.plotly_chart(
            vbar_figure(sd, "DayName", "ridership",
                        f"Average Ridership by Day of Week — {focus_name}", "Day of Week",
                        "Average ridership",
                        colors=[P["accent"] if d in ("Saturday", "Sunday") else P["primary"]
                                for d in sd["DayName"]],
                        height=340),
            width="stretch", config=PLOTLY_CONFIG,
        )

    st.markdown("**5 &middot; Monthly Trend**")
    monthly = pp.resample_series(series, "Monthly")
    if not monthly.empty:
        fig = go.Figure(
            go.Bar(
                x=monthly["period_start"],
                y=monthly["ridership"],
                marker=dict(color=P["primary"]),
                hovertemplate="<b>%{x|%b %Y}</b><br>Total: %{y:,.0f}<extra></extra>",
            )
        )
        fig.update_layout(**base_layout(
            f"Monthly Ridership — {focus_name}", "Month", "Ridership", height=330, hovermode="x"
        ).to_plotly_json())
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


# --------------------------------------------------------------------------
# Tab 4 - Borough & geographic analysis
# --------------------------------------------------------------------------


def render_geographic(v: View) -> None:
    if v.stations.empty:
        st.info("No station complexes match the current filter selection.")
        return

    section("5 &middot; Ridership by Borough",
            "Totals, transfers and station counts for the current selection.")
    if v.boroughs.empty:
        st.info("No borough data available for this selection.")
    else:
        b = v.boroughs.copy()
        b["color"] = b["borough"].map(lambda x: BOROUGH_COLORS.get(x, BOROUGH_FALLBACK))
        fig = go.Figure(
            go.Bar(
                x=b["borough"],
                y=b["Total_Ridership"],
                marker=dict(color=b["color"]),
                customdata=np.stack([b["Stations"], b["Total_Transfers"]], axis=-1),
                hovertemplate=(
                    "<b>%{x}</b><br>Total ridership: %{y:,.0f}<br>"
                    "Station complexes: %{customdata[0]:,.0f}<br>"
                    "Transfers: %{customdata[1]:,.0f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(**base_layout(
            "Ridership by Borough", "Borough", "Total Ridership", height=420
        ).to_plotly_json())
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("**Station complexes per borough**")
            fig_s = go.Figure(
                go.Bar(x=b["borough"], y=b["Stations"], marker=dict(color=b["color"]),
                       hovertemplate="<b>%{x}</b><br>Stations: %{y:,.0f}<extra></extra>")
            )
            fig_s.update_layout(**base_layout(
                "Station Complexes by Borough", "Borough", "Station complexes",
                height=330, hovermode="x",
            ).to_plotly_json())
            st.plotly_chart(fig_s, width="stretch", config=PLOTLY_CONFIG)
        with c2:
            st.markdown("**Transfers by borough**")
            fig_t = go.Figure(
                go.Bar(x=b["borough"], y=b["Total_Transfers"], marker=dict(color=b["color"]),
                       hovertemplate="<b>%{x}</b><br>Transfers: %{y:,.0f}<extra></extra>")
            )
            fig_t.update_layout(**base_layout(
                "Recorded Transfers by Borough", "Borough", "Transfers",
                height=330, hovermode="x",
            ).to_plotly_json())
            st.plotly_chart(fig_t, width="stretch", config=PLOTLY_CONFIG)

        st.markdown("**Borough summary table**")
        st.dataframe(
            b[["borough", "Total_Ridership", "Total_Transfers", "Stations", "Raw_Rows"]],
            hide_index=True, width="stretch",
            column_config={
                "borough": st.column_config.TextColumn("Borough"),
                "Total_Ridership": st.column_config.NumberColumn("Total Ridership", format="%,d"),
                "Total_Transfers": st.column_config.NumberColumn("Transfers", format="%,d"),
                "Stations": st.column_config.NumberColumn("Stations"),
                "Raw_Rows": st.column_config.NumberColumn("Raw API rows"),
            },
        )

    st.divider()
    section(
        "6 &middot; Geographic Analysis",
        "One marker per station complex, positioned with the dataset's own "
        "latitude/longitude and sized by total ridership. Nothing is geocoded "
        "externally; complexes that share coordinates are merged into one marker.",
    )
    points = an.map_points(v.stations)
    if points.empty:
        st.warning("No station coordinates available for this selection.")
        return
    st.plotly_chart(station_map_figure(points), width="stretch", config=PLOTLY_CONFIG)

    merged = points.loc[points["Stations_Merged"] > 1, "station_complex"].tolist()
    if merged:
        info_strip(
            f"{len(merged)} marker(s) combine more than one station complex sharing "
            f"coordinates (for example {', '.join(merged[:3])}). Their volumes are "
            "summed into a single point."
        )

    st.markdown("**Largest stations on the map**")
    st.dataframe(
        points.nlargest(15, "Total_Ridership")[
            ["station_complex", "borough", "Total_Ridership", "Avg_Hourly_Ridership",
             "latitude", "longitude"]
        ].reset_index(drop=True),
        hide_index=True, width="stretch",
        column_config={
            "station_complex": st.column_config.TextColumn("Station Complex", width="large"),
            "borough": st.column_config.TextColumn("Borough"),
            "Total_Ridership": st.column_config.NumberColumn("Total Ridership", format="%,d"),
            "Avg_Hourly_Ridership": st.column_config.NumberColumn("Avg Hourly", format="%.1f"),
            "latitude": st.column_config.NumberColumn("Latitude", format="%.5f"),
            "longitude": st.column_config.NumberColumn("Longitude", format="%.5f"),
        },
    )
    caption(
        "Average Hourly Ridership is the complex's total divided by the number of "
        "hourly timestamps in the selected window, so compare stations within the "
        "same window."
    )


# --------------------------------------------------------------------------
# Tab 5 - Payment & fare
# --------------------------------------------------------------------------


def render_payment_fare(v: View) -> None:
    section("7 &middot; Ridership by Payment Method",
            "Categories are read from the dataset at runtime — nothing is hard-coded.")
    if v.payments.empty:
        st.info("No payment-method data for this selection.")
    else:
        p = v.payments.copy()
        p["share"] = 100.0 * p["Total_Ridership"] / max(p["Total_Ridership"].sum(), 1)
        colors = [P["accent"] if name == "omny" else P["primary"]
                  for name in p["payment_method"]]
        fig = go.Figure(
            go.Bar(
                x=p["payment_method"], y=p["Total_Ridership"], marker=dict(color=colors),
                customdata=np.stack([p["share"], p["Total_Transfers"], p["Raw_Rows"]], axis=-1),
                hovertemplate=(
                    "<b>%{x}</b><br>Ridership: %{y:,.0f}<br>Share: %{customdata[0]:.1f}%<br>"
                    "Transfers: %{customdata[1]:,.0f}<br>Raw rows: %{customdata[2]:,.0f}"
                    "<extra></extra>"
                ),
            )
        )
        fig.update_layout(**base_layout(
            "Ridership by Payment Method", "Payment method", "Total Ridership",
            height=400, hovermode="x",
        ).to_plotly_json())
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

        st.markdown("**Share of ridership**")
        fig_sh = go.Figure(
            go.Pie(
                labels=p["payment_method"], values=p["Total_Ridership"], hole=0.55,
                marker=dict(colors=colors, line=dict(color=P["bg"], width=2)),
                hovertemplate="<b>%{label}</b><br>%{value:,.0f} riders (%{percent})<extra></extra>",
            )
        )
        fig_sh.update_layout(
            title=dict(text="Payment Method Share", font=dict(size=15, color=P["text"]),
                       x=0.01, xanchor="left"),
            height=380, margin=dict(l=10, r=10, t=52, b=10),
            paper_bgcolor=P["paper"], plot_bgcolor=P["bg"], showlegend=True,
            legend=dict(font=dict(size=12, color=P["body"])),
            font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=12, color=P["body"]),
        )
        st.plotly_chart(fig_sh, width="stretch", config=PLOTLY_CONFIG)

    st.divider()
    section("8 &middot; Ridership by Fare Class",
            "Fare categories are read from the dataset at runtime.")
    if v.fares.empty:
        st.info("No fare-class data for this selection.")
    else:
        f = v.fares.sort_values("Total_Ridership", ascending=False).reset_index(drop=True)
        st.plotly_chart(
            hbar_figure(f, "fare_class_category", "Total_Ridership",
                        "Ridership by Fare Class", len(f), color=P["accent"]),
            width="stretch", config=PLOTLY_CONFIG,
        )
        st.dataframe(
            f[["fare_class_category", "Total_Ridership", "Total_Transfers", "Raw_Rows"]],
            hide_index=True, width="stretch",
            column_config={
                "fare_class_category": st.column_config.TextColumn("Fare Class", width="large"),
                "Total_Ridership": st.column_config.NumberColumn("Total Ridership", format="%,d"),
                "Total_Transfers": st.column_config.NumberColumn("Transfers", format="%,d"),
                "Raw_Rows": st.column_config.NumberColumn("Raw API rows"),
            },
        )

    with st.expander("Transit modes present in this selection", expanded=False):
        if v.mode_totals is None or v.mode_totals.empty:
            st.write("No transit-mode data for this selection.")
        else:
            st.dataframe(
                v.mode_totals.rename(columns={
                    "transit_mode": "Transit mode", "ridership": "Total Ridership",
                    "transfers": "Transfers", "raw_rows": "Raw API rows",
                }),
                hide_index=True, width="stretch",
            )


# --------------------------------------------------------------------------
# Tab 6 - Data quality
# --------------------------------------------------------------------------


def render_quality(v: View) -> None:
    section("9 &middot; Data Quality",
            "Measured on the current filter selection, except where a metric is "
            "explicitly labelled as dataset-wide.")

    raw_total = get_raw_row_count()
    kpis = v.kpis

    q1, q2 = st.columns(2)
    q1.metric("Analysis records", fmt_int(kpis["n_observations"]),
              help="Distinct hourly timestamps aggregated for this selection — the rows "
                   "actually held in memory and plotted.")
    q2.metric("Raw dataset size", fmt_compact(raw_total),
              help=f"{fmt_int(raw_total)} rows in the full NYC Open Data dataset. "
                   "Reported for context only; those rows are never loaded.")

    info_strip(
        f"The dashboard never loads the raw dataset. For the current selection it "
        f"requested <b>server-side aggregates</b> from the NYC Open Data API and holds "
        f"<b>{fmt_int(kpis['n_observations'])} hourly records</b> in memory. The raw "
        f"dataset contains <b>{fmt_int(raw_total)}</b> rows — roughly "
        f"<b>{raw_total / max(kpis['n_observations'], 1):,.0f}×</b> the analysed volume."
    )

    st.markdown("**Selection coverage**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records used", fmt_int(kpis["n_observations"]))
    c2.metric("Unique station complexes", fmt_int(kpis["n_stations"]))
    c3.metric("Boroughs", fmt_int(kpis["n_boroughs"]))
    c4.metric("Raw API rows summed", fmt_compact(v.raw_rows),
              help="How many dataset rows the server aggregated to produce these numbers.")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Minimum timestamp", fmt_datetime(kpis["min_timestamp"]))
    c6.metric("Maximum timestamp", fmt_datetime(kpis["max_timestamp"]))
    c7.metric("Total ridership", fmt_compact(kpis["total_ridership"]))
    c8.metric("Total transfers", fmt_compact(kpis["total_transfers"]))

    st.markdown("**Timestamp continuity**")
    if v.hourly.empty:
        st.info("No observations in the current selection.")
    else:
        expected = int(
            (pd.Timestamp(kpis["max_timestamp"]) - pd.Timestamp(kpis["min_timestamp"]))
            .total_seconds()
        ) // 3600 + 1
        observed = int(len(v.hourly))
        duplicates = int(v.hourly["timestamp"].duplicated().sum())
        missing = max(expected - observed, 0)

        d1, d2, d3 = st.columns(3)
        d1.metric("Observed hourly timestamps", fmt_int(observed))
        d2.metric("Expected timestamps", fmt_int(expected),
                  help="Every hour between the first and last observed timestamp.")
        d3.metric("Missing timestamps", fmt_int(missing),
                  help="Hours in the window with no aggregated row.")

        e1, e2 = st.columns(2)
        e1.metric("Duplicate timestamps", fmt_int(duplicates),
                  help="Repeated timestamp keys in the aggregated series. These are summed "
                       "during loading so a duplicate cannot double count.")
        e2.metric("Zero-ridership hours", fmt_int(int((v.hourly["ridership"] == 0).sum())),
                  help="Genuine observations of zero riders — valid values, not missing.")

        if missing:
            info_strip(
                f"<b>{fmt_int(missing)} hourly timestamp(s)</b> are absent between the first "
                "and last observation. They are shown as breaks in the lines and are "
                "<b>not</b> interpolated, filled or estimated.",
                kind="warn",
            )
        else:
            info_strip("The selected window is continuous: no missing hourly timestamps.")

    st.markdown("**Missing values by column (analysis records)**")
    frames = {
        "hourly series": v.hourly,
        "station totals": v.station_totals,
        "borough totals": v.borough_totals,
        "payment totals": v.payment_totals,
        "fare totals": v.fare_totals,
        "station reference": v.meta,
    }
    rows = []
    for label, frame in frames.items():
        if frame is None or frame.empty:
            continue
        for column in frame.columns:
            rows.append({
                "Cube": label, "Column": column, "Rows": int(len(frame)),
                "Missing values": int(frame[column].isna().sum()),
                "Missing %": round(100.0 * float(frame[column].isna().mean()), 3),
            })
    missing_table = pd.DataFrame(rows)
    if missing_table.empty:
        st.info("No aggregate cubes loaded for this selection.")
    else:
        left, right = st.columns([1, 2], gap="medium")
        with left:
            st.dataframe(missing_table, hide_index=True, width="stretch")
        with right:
            worst = missing_table.sort_values("Missing %", ascending=False).head(12)
            if float(worst["Missing %"].max()) <= 0:
                info_strip("No missing values were found in the aggregated cubes.")
            else:
                fig_miss = go.Figure(
                    go.Bar(
                        x=worst["Missing %"], y=worst["Cube"] + " · " + worst["Column"],
                        orientation="h", marker=dict(color=P["accent"]),
                        hovertemplate="<b>%{y}</b><br>Missing: %{x:.3f}%<extra></extra>",
                    )
                )
                fig_miss.update_layout(
                    title=dict(text="Columns with missing values",
                               font=dict(size=15, color=P["text"]), x=0.01, xanchor="left"),
                    xaxis=dict(title="Missing %", gridcolor=P["grid"], zeroline=False,
                               tickfont=dict(color=P["body"])),
                    yaxis=dict(title=None, gridcolor="rgba(0,0,0,0)", automargin=True,
                               tickfont=dict(color=P["body"])),
                    height=max(300, 26 * len(worst) + 120),
                    margin=dict(l=10, r=20, t=50, b=10),
                    plot_bgcolor=P["bg"], paper_bgcolor=P["paper"], showlegend=False,
                    font=dict(family="Source Sans Pro, Segoe UI, sans-serif",
                              size=12, color=P["body"]),
                )
                st.plotly_chart(fig_miss, width="stretch", config=PLOTLY_CONFIG)

    with st.expander("Local aggregate cache (data/processed)", expanded=False):
        st.caption(
            "Every aggregate is memoised as a Parquet file, so a repeated filter "
            "combination never re-queries the API."
        )
        inventory = get_cache_inventory()
        if inventory.empty:
            st.write("The cache is empty.")
        else:
            st.dataframe(inventory, hide_index=True, width="stretch")
            st.caption(
                f"{len(inventory)} cached aggregate(s), "
                f"{float(inventory['Size (KB)'].sum()) / 1024:.2f} MB on disk versus "
                f"{fmt_compact(raw_total)} raw dataset rows."
            )

    with st.expander("Time-series preparation for Phase 2 / Phase 3", expanded=False):
        st.caption(
            "The same pipeline that feeds these charts can export a clean modelling "
            "table. No lag, rolling or model features are computed here — that belongs "
            "to the forecasting phases."
        )
        table = pp.build_forecast_table(v.hourly)
        if table.empty:
            st.write("No series to export.")
        else:
            st.markdown(
                f"`timestamp, total_ridership` — {len(table):,} rows, "
                f"{table['timestamp'].min():%d %b %Y} to "
                f"{table['timestamp'].max():%d %b %Y}"
            )
            st.dataframe(table.head(200), hide_index=True, width="stretch")
        st.markdown(
            "`timestamp, station_complex, borough, ridership` — available from the "
            "station-level loader for panel forecasting."
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def render_header() -> None:
    st.markdown('<div class="app-eyebrow">Time Series Analysis &middot; Phase 1</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{APP_SUBTITLE}</div>', unsafe_allow_html=True)

    lo, hi = dl.dataset_span()
    info_strip(
        f"<b>Source:</b> {DATASET_LABEL} &nbsp;&middot;&nbsp; "
        f"<b>Dataset span:</b> {fmt_date(lo)} &rarr; {fmt_date(hi)} &nbsp;&middot;&nbsp; "
        f"<b>Default analysis period:</b> {fmt_date(dl.DEFAULT_START)} &rarr; "
        f"{fmt_date(dl.DEFAULT_END)} &nbsp;&middot;&nbsp; "
        "<b>Scope:</b> descriptive statistics only — no forecasting in Phase 1"
    )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="\U0001F687",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state.setdefault("dark_mode", False)
    configure_palette()

    try:
        with st.spinner("Loading station reference and dataset categories…"):
            domains = get_domains()
            meta = get_station_meta()
    except DataLoadError as exc:
        st.error(str(exc), icon="\U0001F6A8")
        st.caption(
            "The dashboard reads aggregates from the NYC Open Data API on demand. "
            "Check your network connection and reload the page."
        )
        return
    except Exception as exc:  # noqa: BLE001 - the app must never crash on load
        st.error(
            f"Unexpected error while preparing the dataset: {type(exc).__name__}: {exc}",
            icon="\U0001F6A8",
        )
        return

    render_header()
    sidebar = build_sidebar(meta, domains)

    try:
        with st.spinner(
            "Aggregating from the NYC Open Data API — a new filter combination can "
            "take up to a minute the first time; results are then cached."
        ):
            view = build_view(sidebar)
    except ApiUnavailableError as exc:
        st.error(str(exc), icon="\U0001F6A8")
        st.caption("Try widening the date range or clearing a filter, then reload.")
        return
    except DataLoadError as exc:
        st.error(str(exc), icon="\U0001F6A8")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error while loading data: {type(exc).__name__}: {exc}",
                 icon="\U0001F6A8")
        return

    if view.hourly.empty:
        st.warning(
            "No observations match the current filter selection. Widen the date range "
            "or clear a filter in the sidebar, then reload.",
            icon="\U0001F6A8",
        )
        return

    tabs = st.tabs([
        "Overview",
        "Temporal Analysis",
        "Station Analysis",
        "Geographic Analysis",
        "Payment & Fare Analysis",
        "Data Quality",
    ])
    with tabs[0]:
        render_overview(view)
    with tabs[1]:
        render_temporal(view)
    with tabs[2]:
        render_station(view)
    with tabs[3]:
        render_geographic(view)
    with tabs[4]:
        render_payment_fare(view)
    with tabs[5]:
        render_quality(view)

    st.markdown(
        '<div class="footer-note">Phase 1 exploratory analysis &middot; all values '
        "computed from server-side aggregates of the NYC Open Data API &middot; "
        "forecasting models are out of scope for this phase</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
