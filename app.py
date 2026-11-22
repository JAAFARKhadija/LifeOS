import html
import logging
import streamlit as st
from datetime import date
import pandas as pd
import plotly.express as px
from datetime import timedelta
from modules.activities import add_activity, get_all_activities
from modules.tasks import (
    add_task,
    get_all_tasks,
    update_task_status,
    delete_task
)
from modules.habits import (
    add_habit,
    get_all_habits,
    log_habit,
    get_habit_logs
)
from machine_learning.predict import predict_task_completion
from modules.ai_agent import chat_with_lifeos, LifeOSAIError, is_ai_configured

# Page configuration
st.set_page_config(
    page_title="LifeOS",
    page_icon="🌱",
    layout="wide"
)

logger = logging.getLogger("lifeos.app")

# Quick-start questions shown above the chat
SUGGESTED_QUESTIONS = [
    "What should I study today?",
    "Which tasks are overdue?",
    "Analyze my productivity this week.",
    "How consistent am I with my habits?",
    "Help me organize my upcoming deadlines.",
]


# ==========================================
# THEME SYSTEM (UI ONLY)
# ------------------------------------------
# Everything in this block is presentation code: colour palettes,
# CSS, and small HTML helpers. It never touches the database, the
# ML model or the Gemini agent.
# ==========================================

THEMES = {
    "dark": {
        "scheme": "dark",
        "plotly": "plotly_dark",
        "bg": "#0F151D",
        "surface": "#161E29",
        "input": "#1C2634",
        "border": "#273445",
        "text": "#E6EBF2",
        "muted": "#8F9DB1",
        "accent": "#3CCB93",
        "on-accent": "#06241A",
        "danger": "#F26B70",
        "warn": "#F2B04A",
        "info": "#6EA3F9",
        "success": "#3CCB93",
        "shadow": "0 1px 2px rgba(0, 0, 0, 0.40)",
    },
    "light": {
        "scheme": "light",
        "plotly": "plotly_white",
        "bg": "#F3F5F9",
        "surface": "#FFFFFF",
        "input": "#F7F9FC",
        "border": "#E2E7EF",
        "text": "#172033",
        "muted": "#5A6A7F",
        "accent": "#0E8F63",
        "on-accent": "#FFFFFF",
        "danger": "#C7343A",
        "warn": "#A8610A",
        "info": "#2860C4",
        "success": "#0E8F63",
        "shadow": "0 1px 3px rgba(16, 24, 40, 0.07), 0 1px 2px rgba(16, 24, 40, 0.04)",
    },
}

# One colour per activity type, shared by the overview cards, the
# history badges and the Plotly chart so they always agree.
ACTIVITY_COLORS = {
    "Study": "#5B8DEF",
    "Gym": "#9B7BEA",
    "Volleyball": "#F2A93B",
}

PRIORITY_TONES = {"High": "danger", "Medium": "warn", "Low": "success"}
STATUS_TONES = {"Pending": "info", "Completed": "success"}

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');"
)

BASE_CSS = """
:root {
    --lo-font: 'Plus Jakarta Sans', 'Segoe UI', system-ui, -apple-system, sans-serif;
    --lo-accent-soft: color-mix(in srgb, var(--lo-accent) 16%, transparent);
}

/* ---------- App shell ---------- */
.stApp { background: var(--lo-bg); color: var(--lo-text); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: var(--lo-font);
}
[data-testid="stHeader"] { background: transparent; color: var(--lo-muted); }
footer { visibility: hidden; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 6rem;
}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background: var(--lo-surface);
}
[data-testid="stSidebar"] { border-right: 1px solid var(--lo-border); }

/* ---------- Text ---------- */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 { color: var(--lo-text); }
[data-testid="stMarkdownContainer"] a { color: var(--lo-accent); }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
    color: var(--lo-muted);
    font-weight: 600;
    font-size: 0.85rem;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p { color: var(--lo-muted); }

/* ---------- Header and section titles ---------- */
.lo-hero { display: flex; align-items: center; gap: 14px; }
.lo-logo {
    width: 52px; height: 52px; flex: none;
    display: grid; place-items: center;
    font-size: 1.7rem;
    border-radius: 16px;
    background: var(--lo-accent-soft);
    border: 1px solid var(--lo-border);
}
.lo-brand {
    font-size: 2rem; font-weight: 800; line-height: 1.1;
    letter-spacing: -0.02em; color: var(--lo-text);
}
.lo-tagline { color: var(--lo-muted); font-size: 0.95rem; margin-top: 2px; }
.lo-date {
    display: inline-block; margin-bottom: 8px;
    padding: 4px 12px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600;
    color: var(--lo-muted);
    background: var(--lo-surface);
    border: 1px solid var(--lo-border);
}
.lo-section {
    margin: 2.6rem 0 1rem; padding-top: 1.5rem;
    border-top: 1px solid var(--lo-border);
    scroll-margin-top: 4rem;
}
.lo-section-first { margin-top: 1.6rem; border-top: none; padding-top: 0; }
.lo-section-title {
    font-size: 1.5rem; font-weight: 800;
    letter-spacing: -0.01em; color: var(--lo-text);
}
.lo-section-sub { color: var(--lo-muted); font-size: 0.92rem; margin-top: 2px; }
.lo-sub {
    font-size: 1.02rem; font-weight: 700;
    color: var(--lo-text); margin: 1.6rem 0 0.7rem;
}

/* ---------- Tones (set --t, used by badges, cards, bars) ---------- */
.lo-tone-danger { --t: var(--lo-danger); }
.lo-tone-warn { --t: var(--lo-warn); }
.lo-tone-info { --t: var(--lo-info); }
.lo-tone-success { --t: var(--lo-success); }
.lo-tone-neutral { --t: var(--lo-muted); }

/* ---------- Cards, KPIs ---------- */
.lo-grid {
    display: grid; gap: 14px;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}
.lo-card, .lo-kpi {
    background: var(--lo-surface);
    border: 1px solid var(--lo-border);
    border-radius: 16px;
    box-shadow: var(--lo-shadow);
}
.lo-card { padding: 18px 20px; }
.lo-kpi { display: flex; align-items: center; gap: 14px; padding: 16px 18px; }
.lo-kpi-icon {
    width: 44px; height: 44px; flex: none;
    display: grid; place-items: center;
    font-size: 1.3rem; border-radius: 12px;
    background: color-mix(in srgb, var(--t) 16%, transparent);
}
.lo-kpi-value {
    font-size: 1.75rem; font-weight: 800; line-height: 1.1;
    color: var(--lo-text); font-variant-numeric: tabular-nums;
}
.lo-kpi-label { font-size: 0.86rem; font-weight: 700; color: var(--lo-text); }
.lo-kpi-hint { font-size: 0.78rem; color: var(--lo-muted); }

/* ---------- Reminders ---------- */
.lo-reminders {
    display: grid; gap: 12px;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}
.lo-remind {
    background: var(--lo-surface);
    border: 1px solid var(--lo-border);
    border-left: 4px solid var(--t);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: var(--lo-shadow);
}
.lo-remind-head { font-weight: 700; color: var(--t); }
.lo-remind-item {
    display: flex; flex-wrap: wrap; justify-content: space-between;
    gap: 2px 12px; padding: 7px 0; margin-top: 7px;
    border-top: 1px solid var(--lo-border); font-size: 0.9rem;
}
.lo-item-title { font-weight: 600; color: var(--lo-text); }
.lo-item-when { color: var(--lo-muted); }

/* ---------- Progress ---------- */
.lo-prog-top {
    display: flex; flex-wrap: wrap; align-items: flex-end;
    justify-content: space-between; gap: 6px 16px; margin-bottom: 12px;
}
.lo-prog-pct {
    font-size: 2rem; font-weight: 800; line-height: 1;
    color: var(--lo-text); font-variant-numeric: tabular-nums;
}
.lo-prog-detail { color: var(--lo-muted); font-size: 0.88rem; }
.lo-bar {
    height: 10px; border-radius: 999px; overflow: hidden;
    background: color-mix(in srgb, var(--lo-text) 10%, transparent);
}
.lo-bar > span {
    display: block; height: 100%; border-radius: 999px;
    background: var(--t, var(--lo-accent));
}

/* ---------- Task / activity / habit content ---------- */
.lo-task-title { font-size: 1.02rem; font-weight: 700; color: var(--lo-text); }
.lo-meta {
    display: flex; flex-wrap: wrap; gap: 4px 16px;
    margin-top: 6px; color: var(--lo-muted); font-size: 0.82rem;
}
.lo-field { margin-bottom: 8px; color: var(--lo-muted); font-size: 0.82rem; }
.lo-field-label { display: block; margin-bottom: 3px; }
.lo-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 999px;
    font-size: 0.76rem; font-weight: 700;
    color: var(--t, var(--lo-muted));
    background: color-mix(in srgb, var(--t, var(--lo-muted)) 15%, transparent);
}
.lo-badge-dot {
    color: var(--lo-text);
    background: color-mix(in srgb, var(--lo-text) 8%, transparent);
}
.lo-badge-dot::before {
    content: ""; width: 8px; height: 8px; border-radius: 50%;
    background: var(--t, var(--lo-muted));
}

/* ---------- ML prediction ---------- */
.lo-ml {
    margin-top: 6px; padding: 12px 14px; border-radius: 12px;
    background: color-mix(in srgb, var(--t) 9%, transparent);
    border: 1px solid color-mix(in srgb, var(--t) 28%, transparent);
}
.lo-ml-label { font-size: 0.78rem; font-weight: 700; color: var(--lo-muted); }
.lo-ml-msg { font-weight: 600; font-size: 0.92rem; color: var(--lo-text); margin: 3px 0 9px; }
.lo-ml-score { font-size: 0.8rem; color: var(--lo-muted); margin-top: 7px; }

/* ---------- Empty state ---------- */
.lo-empty {
    display: flex; align-items: center; gap: 12px;
    padding: 18px 20px; border-radius: 16px;
    border: 1px dashed var(--lo-border);
    background: var(--lo-surface); color: var(--lo-muted);
}
.lo-empty-icon { font-size: 1.4rem; }

/* ---------- Sidebar ---------- */
.lo-side-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
.lo-side-brand .lo-logo { width: 38px; height: 38px; font-size: 1.2rem; border-radius: 12px; }
.lo-side-name { font-size: 1.2rem; font-weight: 800; color: var(--lo-text); }
.lo-nav a {
    display: block; padding: 9px 12px; margin-bottom: 2px;
    border-radius: 10px; font-weight: 600; font-size: 0.92rem;
    color: var(--lo-text); text-decoration: none;
}
.lo-nav a:hover { background: var(--lo-accent-soft); }

/* ---------- Native widgets ---------- */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 10px; font-weight: 600;
    border: 1px solid var(--lo-border);
    background: var(--lo-surface); color: var(--lo-text);
    transition: border-color 0.15s, color 0.15s;
}
.stButton > button p, [data-testid="stFormSubmitButton"] > button p { color: inherit; }
.stButton > button:hover { border-color: var(--lo-accent); color: var(--lo-accent); }
[data-testid="stFormSubmitButton"] > button {
    background: var(--lo-accent); border-color: var(--lo-accent);
    color: var(--lo-on-accent);
}
[data-testid="stFormSubmitButton"] > button:hover {
    color: var(--lo-on-accent); filter: brightness(1.08);
}
.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible {
    outline: 2px solid var(--lo-accent); outline-offset: 2px;
}

[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
[data-baseweb="select"] > div {
    background-color: var(--lo-input) !important;
    border-color: var(--lo-border) !important;
    border-radius: 10px !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--lo-accent) !important;
}
input, textarea { color: var(--lo-text) !important; }
input::placeholder, textarea::placeholder { color: var(--lo-muted) !important; opacity: 0.85; }
[data-baseweb="select"] div, [data-baseweb="select"] span { color: var(--lo-text) !important; }
[data-baseweb="select"] svg { fill: var(--lo-muted); }
[data-testid="stNumberInput"] button {
    background: var(--lo-input) !important; color: var(--lo-text) !important;
    border-color: var(--lo-border) !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--lo-border); border-radius: 14px;
    background: var(--lo-surface); box-shadow: var(--lo-shadow);
    overflow: hidden;
}
[data-testid="stExpander"] details { border: none !important; background: transparent; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span { color: var(--lo-text); }
[data-testid="stExpander"] summary p { font-weight: 700; }
[data-testid="stForm"] {
    border: none !important; background: transparent !important;
    padding: 0.25rem 0 0 !important;
}

[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stAlert"] p { color: var(--lo-text); }

[data-testid="stPlotlyChart"] {
    background: var(--lo-surface);
    border: 1px solid var(--lo-border);
    border-radius: 16px; padding: 10px 12px;
    box-shadow: var(--lo-shadow);
}

[data-testid="stChatMessage"] {
    background: var(--lo-surface);
    border: 1px solid var(--lo-border);
    border-radius: 16px; padding: 14px 16px;
    box-shadow: var(--lo-shadow);
}
[data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: var(--lo-bg) !important; }
[data-testid="stChatInput"] {
    background: var(--lo-surface) !important;
    border: 1px solid var(--lo-border); border-radius: 14px;
}

/* ---------- Bordered containers used as cards ----------
   st.container(border=True) has no styling hook, so each card holds an
   invisible marker. The rule targets only the innermost bordered block
   that holds a marker. If Streamlit changes its markup this simply stops
   matching and the cards fall back to Streamlit's own bordered container. */
[data-testid="stVerticalBlockBorderWrapper"]:has(.lo-card-marker):not(:has([data-testid="stVerticalBlockBorderWrapper"] .lo-card-marker)) {
    background: var(--lo-surface) !important;
    border: 1px solid var(--lo-border) !important;
    border-radius: 16px !important;
    box-shadow: var(--lo-shadow);
    padding: 8px 12px;
}
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] .lo-card-marker),
[data-testid="element-container"]:has(> [data-testid="stMarkdown"] .lo-card-marker),
.element-container:has(> .stMarkdown .lo-card-marker) { display: none; }

/* ---------- Responsive and motion ---------- */
@media (max-width: 640px) {
    .lo-brand { font-size: 1.55rem; }
    .lo-section-title { font-size: 1.25rem; }
    .block-container, [data-testid="stMainBlockContainer"] {
        padding-left: 1rem; padding-right: 1rem;
    }
}
@media (prefers-reduced-motion: reduce) {
    .stButton > button, [data-testid="stFormSubmitButton"] > button { transition: none; }
}
"""


def build_theme_css(mode):
    """Return the <style> block (colour variables + shared CSS) for a theme."""
    palette = THEMES[mode]
    variables = "".join(
        f"--lo-{key}:{value};"
        for key, value in palette.items()
        if key not in ("scheme", "plotly")
    )
    return (
        f"<style>{FONT_IMPORT}"
        f":root{{color-scheme:{palette['scheme']};{variables}}}"
        f"{BASE_CSS}</style>"
    )


def esc(value):
    """Escape database text before it is placed inside HTML."""
    return html.escape(str(value))


def render_html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def card_marker():
    """Invisible marker that lets CSS style the surrounding st.container."""
    render_html('<span class="lo-card-marker"></span>')


def section_header(icon, title, subtitle, anchor, first=False):
    css_class = "lo-section lo-section-first" if first else "lo-section"
    subtitle_html = (
        f'<div class="lo-section-sub">{esc(subtitle)}</div>' if subtitle else ""
    )
    render_html(
        f'<div class="{css_class}" id="{anchor}">'
        f'<div class="lo-section-title" role="heading" aria-level="2">'
        f'{icon} {esc(title)}</div>{subtitle_html}</div>'
    )


def sub_header(text):
    render_html(
        f'<div class="lo-sub" role="heading" aria-level="3">{esc(text)}</div>'
    )


def empty_state(icon, message):
    render_html(
        f'<div class="lo-empty"><span class="lo-empty-icon" aria-hidden="true">'
        f'{icon}</span><span>{esc(message)}</span></div>'
    )


def badge(text, tone="neutral", color=None, dot=False):
    style = f' style="--t:{color}"' if color else ""
    dot_class = " lo-badge-dot" if dot else ""
    return (
        f'<span class="lo-badge lo-tone-{tone}{dot_class}"{style}>{esc(text)}</span>'
    )


def kpi_card(icon, label, value, hint, tone="info", color=None):
    style = f' style="--t:{color}"' if color else ""
    return (
        f'<div class="lo-kpi lo-tone-{tone}"{style}>'
        f'<div class="lo-kpi-icon" aria-hidden="true">{icon}</div>'
        f'<div><div class="lo-kpi-value">{esc(value)}</div>'
        f'<div class="lo-kpi-label">{esc(label)}</div>'
        f'<div class="lo-kpi-hint">{esc(hint)}</div></div></div>'
    )


def reminder_card(tone, heading, rows):
    """rows is a list of (title, when) pairs; empty list gives a heading-only card."""
    items = "".join(
        f'<div class="lo-remind-item"><span class="lo-item-title">{esc(title_text)}'
        f'</span><span class="lo-item-when">{esc(when_text)}</span></div>'
        for title_text, when_text in rows
    )
    return (
        f'<div class="lo-remind lo-tone-{tone}">'
        f'<div class="lo-remind-head">{esc(heading)}</div>{items}</div>'
    )


def progress_bar(fraction, tone="success", label="Progress"):
    percent = round(max(0.0, min(1.0, float(fraction))) * 100)
    return (
        f'<div class="lo-bar lo-tone-{tone}" role="progressbar" aria-label="{esc(label)}" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
        f'<span style="width:{percent}%"></span></div>'
    )


def style_chart(fig, mode, week_start, week_end):
    """Apply the active theme to the weekly Plotly chart (no data changes)."""
    palette = THEMES[mode]
    fig.update_layout(
        template=palette["plotly"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Plus Jakarta Sans, Segoe UI, system-ui, sans-serif",
            color=palette["text"],
            size=13,
        ),
        title=dict(
            x=0, xanchor="left", y=0.95, yref="container",
            font=dict(size=16, color=palette["text"]),
        ),
        margin=dict(l=8, r=8, t=84, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(color=palette["muted"]),
        ),
        bargap=0.28,
        bargroupgap=0.06,
        hoverlabel=dict(
            bgcolor=palette["input"],
            bordercolor=palette["border"],
            font=dict(color=palette["text"]),
        ),
    )
    fig.update_xaxes(
        type="date",
        range=[
            pd.Timestamp(week_start) - pd.Timedelta(hours=12),
            pd.Timestamp(week_end) + pd.Timedelta(hours=12),
        ],
        dtick=86400000,
        tickformat="%a %d",
        showgrid=False,
        linecolor=palette["border"],
        tickfont=dict(color=palette["muted"]),
        title_font=dict(color=palette["muted"]),
    )
    fig.update_yaxes(
        rangemode="tozero",
        ticksuffix=" h",
        gridcolor=palette["border"],
        zeroline=False,
        tickfont=dict(color=palette["muted"]),
        title_font=dict(color=palette["muted"]),
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>%{x|%A %d %b}<br>"
            "%{y:.1f} h<extra></extra>"
        )
    )
    return fig


def toggle_theme():
    """Button callback: flip between dark and light (UI state only)."""
    current = st.session_state.get("lifeos_theme", "dark")
    st.session_state["lifeos_theme"] = "light" if current == "dark" else "dark"


# Preserve the selected theme across reruns
if "lifeos_theme" not in st.session_state:
    st.session_state["lifeos_theme"] = "dark"

theme_mode = st.session_state["lifeos_theme"]
render_html(build_theme_css(theme_mode))


# ==========================================
# SIDEBAR (quick navigation to page sections)
# ==========================================

with st.sidebar:
    render_html(
        '<div class="lo-side-brand"><div class="lo-logo" aria-hidden="true">🌱</div>'
        '<div class="lo-side-name">LifeOS</div></div>'
        '<nav class="lo-nav" aria-label="Sections">'
        '<a href="#lo-dashboard" target="_self">📊 Dashboard</a>'
        '<a href="#lo-tasks" target="_self">📚 Study Tasks</a>'
        '<a href="#lo-activities" target="_self">🏃 Activities</a>'
        '<a href="#lo-habits" target="_self">🎯 Habits</a>'
        '<a href="#lo-assistant" target="_self">🤖 AI Assistant</a>'
        '<a href="#lo-summary" target="_self">✨ Daily Summary</a>'
        '</nav>'
    )


# Header
header_left, header_right = st.columns([5, 2])

with header_left:
    render_html(
        '<div class="lo-hero"><div class="lo-logo" aria-hidden="true">🌱</div>'
        '<div><div class="lo-brand" role="heading" aria-level="1">LifeOS</div>'
        '<div class="lo-tagline">Your Personal AI Assistant</div></div></div>'
    )

with header_right:
    render_html(
        f'<div class="lo-date">{esc(date.today().strftime("%A, %d %B %Y"))}</div>'
    )
    st.button(
        "☀️ Switch to Light" if theme_mode == "dark" else "🌙 Switch to Dark",
        key="lifeos_theme_button",
        on_click=toggle_theme,
        use_container_width=True
    )


# ==========================================
# LIFEOS DASHBOARD
# ==========================================

section_header(
    "📊",
    "Your Personal Dashboard",
    "Your tasks, deadlines and activity at a glance.",
    "lo-dashboard",
    first=True
)

# Retrieve database information
dashboard_tasks = get_all_tasks()
dashboard_activities = get_all_activities()

# ------------------------------------------
# Calculations (unchanged from the original)
# ------------------------------------------

today = date.today()

overdue_tasks = []
due_today_tasks = []
upcoming_tasks = []

for task in dashboard_tasks:

    task_id = task[0]
    title = task[1]
    due_date = task[5]
    status = task[7]

    if status != "Pending" or not due_date:
        continue

    deadline = date.fromisoformat(str(due_date)[:10])

    if deadline < today:
        overdue_tasks.append((title, deadline))

    elif deadline == today:
        due_today_tasks.append((title, deadline))

    elif (deadline - today).days <= 3:
        upcoming_tasks.append((title, deadline))

# Task statistics
total_tasks = len(dashboard_tasks)

completed_tasks = sum(
    1 for task in dashboard_tasks
    if task[7] == "Completed"
)

pending_tasks = sum(
    1 for task in dashboard_tasks
    if task[7] == "Pending"
)

# Activity statistics (duration is in minutes)
study_minutes = sum(
    activity[3] for activity in dashboard_activities
    if activity[1] == "Study"
)

gym_minutes = sum(
    activity[3] for activity in dashboard_activities
    if activity[1] == "Gym"
)

volleyball_minutes = sum(
    activity[3] for activity in dashboard_activities
    if activity[1] == "Volleyball"
)

# Convert minutes to hours
study_hours = round(study_minutes / 60, 1)
gym_hours = round(gym_minutes / 60, 1)
volleyball_hours = round(volleyball_minutes / 60, 1)

# ------------------------------------------
# KPI cards
# ------------------------------------------

render_html(
    '<div class="lo-grid">'
    + kpi_card("📋", "Total Tasks", total_tasks, "All tasks in your list", "info")
    + kpi_card("✅", "Completed", completed_tasks, "Tasks marked as done", "success")
    + kpi_card("⏳", "Pending", pending_tasks, "Tasks still to do", "warn")
    + kpi_card("🏃", "Total Activities", len(dashboard_activities),
               "Sessions recorded", "neutral")
    + '</div>'
)

# ==========================================
# AUTOMATED TASK REMINDERS
# ==========================================

sub_header("🔔 Automatic Task Reminders")

reminder_cards = []

if overdue_tasks:
    reminder_cards.append(reminder_card(
        "danger",
        f"🚨 {len(overdue_tasks)} overdue task(s)",
        [(title_text, f"Overdue since {deadline_value}")
         for title_text, deadline_value in overdue_tasks]
    ))

if due_today_tasks:
    reminder_cards.append(reminder_card(
        "warn",
        f"⏰ {len(due_today_tasks)} task(s) due today",
        [(title_text, "Due today")
         for title_text, deadline_value in due_today_tasks]
    ))

if upcoming_tasks:
    reminder_cards.append(reminder_card(
        "info",
        f"📅 {len(upcoming_tasks)} upcoming deadline(s)",
        [(title_text, f"Due {deadline_value}")
         for title_text, deadline_value in upcoming_tasks]
    ))

if not overdue_tasks and not due_today_tasks and not upcoming_tasks:
    reminder_cards.append(reminder_card(
        "success",
        "✅ You're all clear! No urgent deadlines.",
        []
    ))

render_html('<div class="lo-reminders">' + "".join(reminder_cards) + '</div>')

# ------------------------------------------
# Study overview
# ------------------------------------------

sub_header("📚 Study Overview")

completion_ratio = (completed_tasks / total_tasks) if total_tasks else 0
completion_percent = round(completion_ratio * 100)

if total_tasks:
    completion_detail = (
        f"{completed_tasks} completed, {pending_tasks} pending "
        f"out of {total_tasks} tasks"
    )
else:
    completion_detail = "No tasks yet"

render_html(
    '<div class="lo-card">'
    '<div class="lo-prog-top">'
    f'<div><div class="lo-prog-pct">{completion_percent}%</div>'
    '<div class="lo-kpi-hint">of your tasks are completed</div></div>'
    f'<div class="lo-prog-detail">{esc(completion_detail)}</div></div>'
    + progress_bar(completion_ratio, "success", "Task completion")
    + '</div>'
)

# ------------------------------------------
# Activity overview
# ------------------------------------------

sub_header("🏃 Activity Overview")

render_html(
    '<div class="lo-grid">'
    + kpi_card("📖", "Study Hours", f"{study_hours} h",
               "Total time logged", color=ACTIVITY_COLORS["Study"])
    + kpi_card("🏋️", "Gym Hours", f"{gym_hours} h",
               "Total time logged", color=ACTIVITY_COLORS["Gym"])
    + kpi_card("🏐", "Volleyball Hours", f"{volleyball_hours} h",
               "Total time logged", color=ACTIVITY_COLORS["Volleyball"])
    + '</div>'
)


# ==========================================
# WEEKLY ACTIVITY CHART
# ==========================================

sub_header("📈 Weekly Activity Overview")

today = date.today()
start_date = today - timedelta(days=6)

weekly_records = []

for activity in dashboard_activities:

    activity_date = date.fromisoformat(activity[4])

    if start_date <= activity_date <= today:

        weekly_records.append({
            "Date": activity_date,
            "Activity": activity[1],
            "Hours": activity[3] / 60
        })

if weekly_records:

    weekly_df = pd.DataFrame(weekly_records)

    chart_data = (
        weekly_df
        .groupby(["Date", "Activity"], as_index=False)["Hours"]
        .sum()
    )

    fig = px.bar(
        chart_data,
        x="Date",
        y="Hours",
        color="Activity",
        barmode="group",
        title="Hours spent per activity (Last 7 Days)",
        labels={
            "Hours": "Duration (hours)",
            "Date": "Day"
        },
        color_discrete_map=ACTIVITY_COLORS
    )

    fig.update_layout(
        xaxis_title="Day",
        yaxis_title="Hours",
        legend_title="Activity"
    )

    style_chart(fig, theme_mode, start_date, today)

    # theme=None: use our own colours instead of Streamlit's chart theme
    st.plotly_chart(fig, use_container_width=True, theme=None)

else:
    empty_state("📭", "No activities recorded during the last 7 days.")


# ==========================================
# STUDY TASK MANAGER
# ==========================================

section_header(
    "📚",
    "Study Task Manager",
    "Plan, track and complete your study work.",
    "lo-tasks"
)

# Add task form
with st.expander("➕ Add a New Task", expanded=True):

    with st.form("add_task_form"):

        title = st.text_input("Task Title")
        subject = st.text_input("Subject")
        task_type = st.selectbox("Task Type",
        ["Cours", "TD", "TP", "Projects", "Series", "Tests"])
        
        col1, col2, col3 = st.columns(3)

        with col1:
            due_date = st.date_input("Deadline", value=date.today())

        with col2:
            priority = st.selectbox(
                "Priority",
                ["Low", "Medium", "High"]
            )

        with col3:
            duration = st.number_input(
                "Estimated Duration (minutes)",
                min_value=0,
                value=60
            )

        submitted = st.form_submit_button("Add Task")

        if submitted:
            if title.strip():
                add_task(
                    title=title.strip(),
                    subject=subject.strip() or None,
                    task_type=task_type,
                    due_date=str(due_date),
                    priority=priority,
                    estimated_duration=duration
                )

                st.success("Task added successfully!")
                st.rerun()

            else:
                st.error("Please enter a task title.")


# Display tasks
sub_header("📋 My Tasks")

tasks = get_all_tasks()

if not tasks:
    empty_state("📝", "No tasks yet. Add your first task above!")

else:
    for task in tasks:

        task_id = task[0]
        title = task[1]
        subject = task[2]
        task_type = task[3]
        created_date = task[4]
        due_date = task[5]
        priority = task[6]
        status = task[7]
        duration = task[8]

        with st.container(border=True):

            card_marker()

            col1, col2, col3 = st.columns([4, 2, 2])

            with col1:
                render_html(
                    f'<div class="lo-task-title">{esc(title)}</div>'
                    '<div class="lo-meta">'
                    f'<span>Subject: {esc(subject or "N/A")}</span>'
                    f'<span>Type: {esc(task_type)}</span>'
                    f'<span>Created: {esc(created_date)}</span>'
                    f'<span>Deadline: {esc(due_date)}</span></div>'
                )

            with col2:
                render_html(
                    '<div class="lo-field"><span class="lo-field-label">Priority</span>'
                    + badge(priority, PRIORITY_TONES.get(priority, "neutral"))
                    + '</div>'
                    f'<div class="lo-field">Duration: {esc(duration or 0)} min</div>'
                )

            with col3:
                render_html(
                    '<div class="lo-field"><span class="lo-field-label">Status</span>'
                    + badge(status, STATUS_TONES.get(status, "neutral"))
                    + '</div>'
                )

                if status == "Pending":
                    if st.button(
                        "✅ Complete",
                        key=f"complete_{task_id}"
                    ):
                        update_task_status(task_id, "Completed")
                        st.rerun()

                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{task_id}"
                ):
                    delete_task(task_id)
                    st.rerun()

            # ==========================================
            # ML TASK COMPLETION PREDICTION
            # ==========================================

            if status == "Pending" and created_date and due_date and duration is not None:

                try:
                    result = predict_task_completion(
                        subject=subject,
                        task_type=task_type,
                        priority=priority,
                        estimated_duration=duration,
                        created_date=created_date,
                        due_date=due_date
                    )

                    probability = result["on_time_probability"]

                    if result["prediction"] == 1:
                        ml_tone = "success"
                        ml_message = "✅ Model estimates likely on-time completion"
                    else:
                        ml_tone = "warn"
                        ml_message = "⚠️ Model flags potential delay"

                    render_html(
                        f'<div class="lo-ml lo-tone-{ml_tone}">'
                        '<div class="lo-ml-label">ML Prediction (Experimental)</div>'
                        f'<div class="lo-ml-msg">{esc(ml_message)}</div>'
                        + progress_bar(probability, ml_tone, "Estimated on-time score")
                        + f'<div class="lo-ml-score">Estimated on-time score: '
                        f'{probability:.1%}</div></div>'
                    )

                except Exception as e:
                    st.error(f"Prediction unavailable: {e}")



# ==========================================
# ACTIVITIES MANAGER
# ==========================================

section_header(
    "🏃",
    "Activities Manager",
    "Log your study, gym and volleyball sessions.",
    "lo-activities"
)

with st.expander("➕ Record a New Activity", expanded=False):

    with st.form("activity_form"):

        activity_type = st.selectbox(
            "Activity Type",
            ["Study", "Gym", "Volleyball"]
        )

        title = st.text_input("Activity Title")

        col1, col2 = st.columns(2)

        with col1:
            duration = st.number_input(
                "Duration (minutes)",
                min_value=1,
                value=60
            )

        with col2:
            activity_date = st.date_input(
                "Activity Date",
                value=date.today(),
                key="activity_date"
            )

        notes = st.text_area("Notes (optional)")

        submitted = st.form_submit_button("Save Activity")

        if submitted:
            if title.strip():
                add_activity(
                    activity_type,
                    title.strip(),
                    duration,
                    str(activity_date),
                    notes.strip() or None
                )

                st.success("Activity saved!")
                st.rerun()
            else:
                st.error("Please enter an activity title.")

# Display activities
sub_header("📊 Activity History")

activities = get_all_activities()

if activities:

    for activity in activities:

        activity_id, activity_type, title, duration, activity_date, notes = activity

        with st.container(border=True):
            card_marker()

            render_html(
                f'<div class="lo-task-title">{esc(title)} '
                + badge(
                    activity_type,
                    color=ACTIVITY_COLORS.get(activity_type),
                    dot=True
                )
                + '</div>'
                '<div class="lo-meta">'
                f'<span>📅 {esc(activity_date)}</span>'
                f'<span>⏱️ {esc(duration)} minutes</span></div>'
            )

            if notes:
                st.caption(notes)

else:
    empty_state("🏃", "No activities recorded yet.")


# ==========================================
# HABIT TRACKER
# ==========================================

section_header(
    "🎯",
    "Daily Habit Tracker",
    "Mark off today's habits and keep your streak going.",
    "lo-habits"
)

today = str(date.today())

# Add a new habit
with st.expander("➕ Create a New Habit"):

    with st.form("habit_form"):

        habit_name = st.text_input("Habit Name")
        description = st.text_input("Description")

        submitted = st.form_submit_button("Create Habit")

        if submitted:
            if habit_name.strip():
                try:
                    add_habit(
                        habit_name.strip(),
                        description.strip() or None
                    )
                    st.success("Habit created!")
                    st.rerun()

                except Exception as error:
                    st.error(f"Could not create habit: {error}")
            else:
                st.warning("Enter a habit name.")

# Display habits
habits = get_all_habits()

sub_header("My Daily Habits")

if not habits:
    empty_state("🎯", "No habits created yet.")

else:
    for habit in habits:

        habit_id = habit[0]
        name = habit[1]
        description = habit[2]

        logs = get_habit_logs(habit_id)

        completed_today = any(
            log[2] == today and log[3] == 1
            for log in logs
        )

        with st.container(border=True):

            card_marker()

            col1, col2 = st.columns([3, 2])

            with col1:
                render_html(f'<div class="lo-task-title">{esc(name)}</div>')
                if description:
                    st.caption(description)

            with col2:

                if completed_today:
                    render_html(
                        '<div class="lo-field">'
                        + badge("Completed today! ✅", "success")
                        + '</div>'
                    )

                    if st.button(
                        "Undo completion",
                        key=f"undo_{habit_id}"
                    ):
                        log_habit(habit_id, today, False)
                        st.rerun()

                else:
                    if st.button(
                        "Mark as completed",
                        key=f"complete_habit_{habit_id}"
                    ):
                        log_habit(habit_id, today, True)
                        st.rerun()

# ==========================================
# LIFEOS AI ASSISTANT
# ==========================================

section_header("🤖", "LifeOS AI Assistant", None, "lo-assistant")

st.caption(
    "Ask questions about your study tasks, activities, "
    "habits and productivity."
)

if not is_ai_configured():
    st.warning(
        "The AI assistant is not configured yet. Add GEMINI_API_KEY to "
        "the .env file and restart LifeOS."
    )

# Initialize chat history
if "lifeos_chat_history" not in st.session_state:
    st.session_state.lifeos_chat_history = []

# Suggested questions (clicking one behaves like typing it)
suggested_prompt = None

suggestion_columns = st.columns(3)

for index, question in enumerate(SUGGESTED_QUESTIONS):
    with suggestion_columns[index % 3]:
        if st.button(
            question,
            key=f"suggested_question_{index}",
            use_container_width=True
        ):
            suggested_prompt = question

# Display previous messages
for item in st.session_state.lifeos_chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

# Chat input
typed_prompt = st.chat_input("Ask LifeOS something...")
prompt = typed_prompt or suggested_prompt

if prompt:

    # Save and display user message
    st.session_state.lifeos_chat_history.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):

        answer_is_error = False

        with st.spinner("LifeOS is thinking..."):

            try:
                answer = chat_with_lifeos(
                    prompt,
                    st.session_state.lifeos_chat_history[:-1]
                )

            except LifeOSAIError as error:
                # Message is already safe to show to the user
                answer = f"⚠️ {error}"
                answer_is_error = True

            except Exception:
                # Full traceback goes to the server log, not to the user
                logger.exception("Unexpected error in the AI assistant")
                answer = (
                    "⚠️ Something unexpected went wrong. "
                    "Please try again."
                )
                answer_is_error = True

        st.markdown(answer)

    # Save assistant response (errors are flagged so they are shown in
    # the chat but not sent back to the model as conversation context)
    assistant_message = {
        "role": "assistant",
        "content": answer
    }

    if answer_is_error:
        assistant_message["error"] = True

    st.session_state.lifeos_chat_history.append(assistant_message)

# ==========================================
# AUTOMATED DAILY PRODUCTIVITY SUMMARY
# ==========================================

section_header(
    "📊",
    "Daily Productivity Summary",
    "A short report built from your current LifeOS data.",
    "lo-summary"
)

if st.button("✨ Generate Today's Summary"):

    with st.spinner("Analyzing your LifeOS data..."):

        try:
            summary = chat_with_lifeos(
                "Generate a concise daily productivity report using my "
                "current LifeOS database. Include pending and overdue "
                "tasks, upcoming deadlines, recent study/gym/volleyball "
                "activities, and recent habit progress. "
                "Give me three practical priorities. "
                "Clearly distinguish missing data from zero activity."
            )

            st.session_state.daily_productivity_summary = summary

        except LifeOSAIError as error:
            st.error(f"Could not generate summary: {error}")

        except Exception:
            logger.exception("Unexpected error while generating the summary")
            st.error("Could not generate the summary. Please try again.")

if "daily_productivity_summary" in st.session_state:
    with st.container(border=True):
        card_marker()
        st.markdown(st.session_state.daily_productivity_summary)