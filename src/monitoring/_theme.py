"""
Shared Theme & Styling Utilities — LoanScope Institutional Design System
========================================================================
Single source of truth for the LoanScope dashboard's visual identity.
Transforms standard Streamlit into a quantitative institutional terminal
(Palantir / Bloomberg / Aladdin aesthetic) with deep slate styling,
monospaced financial tabular numerals, and custom Plotly dark palettes.

Contains presentation helpers only. No business logic or metrics.
"""

from typing import Optional
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Institutional Color Palette (Aligned with Signal Design System)
# ---------------------------------------------------------------------------
COLORS = {
    "bg_dark": "#070B14",
    "bg_card": "#0F172A",
    "bg_card_alt": "#141E33",
    "border": "#1E293B",
    "border_gold": "rgba(200, 168, 112, 0.4)",
    "gold": "#C8A870",
    "cyan": "#38BDF8",
    "blue": "#2563EB",
    "green": "#22C55E",
    "red": "#EF4444",
    "amber": "#F59E0B",
    "purple": "#A855F7",
    "slate_text": "#F8FAFC",
    "muted_text": "#94A3B8",
    "subtle_text": "#64748B",
    # Backward compatibility
    "base": "#94A3B8",
    "adverse": "#EF4444",
    "favorable": "#38BDF8",
    "warn": "#F59E0B",
    "pass": "#22C55E",
    "fail": "#EF4444",
}

STATUS_COLOR_MAP = {"PASS": COLORS["pass"], "WARN": COLORS["warn"], "FAIL": COLORS["fail"]}


def inject_theme() -> None:
    """
    Inject institutional CSS styling, Google Fonts, and custom widget tokens.
    Call once per page, first.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Global Canvas Defaults - Enforce Deep Midnight Navy Terminal */
        html, body, .stApp {
            background-color: #070B14 !important;
            color: #F8FAFC !important;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
            -webkit-font-smoothing: antialiased;
        }

        /* Typography Hierarchy */
        h1, .main-header {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: #F8FAFC !important;
            margin-bottom: 0.3rem !important;
        }
        h2 {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em !important;
            color: #F1F5F9 !important;
            margin-top: 1.2rem !important;
        }
        h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            color: #E2E8F0 !important;
        }
        h4, h5, h6 {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 600 !important;
            color: #CBD5E1 !important;
        }
        .sub-header {
            font-size: 1.0rem !important;
            color: #94A3B8 !important;
            margin-bottom: 1.2rem !important;
            line-height: 1.5 !important;
        }

        /* Institutional Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0A0F1D !important;
            border-right: 1px solid #1E293B !important;
        }
        section[data-testid="stSidebar"] .stMarkdown p {
            color: #94A3B8 !important;
        }

        /* High-End Quantitative Metric Cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #0F172A 0%, #0B1220 100%) !important;
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
            padding: 14px 18px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        div[data-testid="stMetric"]:hover {
            border-color: rgba(200, 168, 112, 0.45) !important;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35), 0 0 12px rgba(200, 168, 112, 0.12) !important;
            transform: translateY(-2px);
        }
        div[data-testid="stMetricLabel"] p {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: #94A3B8 !important;
            margin-bottom: 4px !important;
        }
        div[data-testid="stMetricValue"] div {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            color: #F8FAFC !important;
            letter-spacing: -0.02em !important;
            font-variant-numeric: tabular-nums !important;
        }
        div[data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
        }

        /* Container & Card Grids */
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            background-color: #0F172A !important;
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
            transition: border-color 0.2s ease !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
            border-color: rgba(200, 168, 112, 0.35) !important;
        }

        /* Module Feature Cards */
        .feature-card {
            background: linear-gradient(180deg, #0F172A 0%, #0A101D 100%);
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 1.3rem;
            margin-bottom: 1rem;
            transition: all 0.2s ease;
        }
        .feature-card:hover {
            border-color: #C8A870;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
            transform: translateY(-2px);
        }
        .feature-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.12rem;
            font-weight: 600;
            color: #38BDF8;
            margin-bottom: 0.4rem;
        }
        .feature-card p {
            color: #94A3B8;
            font-size: 0.92rem;
            line-height: 1.5;
            margin-bottom: 0.8rem;
        }

        /* Status & Category Badge Chips */
        .badge-chip {
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 9999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.04em;
        }
        .badge-gold {
            background-color: rgba(200, 168, 112, 0.12);
            color: #C8A870;
            border: 1px solid rgba(200, 168, 112, 0.3);
        }
        .badge-cyan {
            background-color: rgba(56, 189, 248, 0.12);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .badge-green {
            background-color: rgba(34, 197, 94, 0.12);
            color: #4ADE80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .badge-red {
            background-color: rgba(239, 68, 68, 0.12);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .badge-amber {
            background-color: rgba(245, 158, 11, 0.12);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        /* Institutional Tabs */
        button[data-baseweb="tab"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            padding: 10px 18px !important;
            border-radius: 6px 6px 0 0 !important;
            color: #94A3B8 !important;
            background: transparent !important;
            transition: all 0.18s ease !important;
            border-bottom: 2px solid transparent !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #F8FAFC !important;
            background: rgba(255, 255, 255, 0.03) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #C8A870 !important;
            border-bottom: 2px solid #C8A870 !important;
            background: rgba(200, 168, 112, 0.06) !important;
        }

        /* Dataframe Modernization */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
            background-color: #0F172A !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
        }

        /* Input Controls & Dropdowns */
        div[data-baseweb="select"] {
            background-color: #0F172A !important;
            border-radius: 6px !important;
        }

        /* Horizontal Divider */
        hr {
            border-color: #1E293B !important;
            margin: 1.5rem 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_mermaid_diagram(graph_definition: str, height: int = 560) -> None:
    """
    Render a real Mermaid.js flowchart (actual arrows/merge lines, not text
    approximations) inside a Streamlit components iframe. Loads mermaid from
    a CDN client-side — no Python package, so nothing new to install on
    Streamlit Cloud. Call with a raw mermaid graph definition string.
    """
    import streamlit.components.v1 as components

    html = f"""
    <div class="mermaid">
    {graph_definition}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'base',
            themeVariables: {{
                background: '#0B1220',
                primaryColor: '#0F172A',
                primaryTextColor: '#F1F5F9',
                primaryBorderColor: '#475569',
                lineColor: '#94A3B8',
                secondaryColor: '#0F172A',
                tertiaryColor: '#0F172A',
                fontFamily: 'Plus Jakarta Sans, sans-serif',
                fontSize: '15px'
            }},
            flowchart: {{ curve: 'basis', padding: 12 }}
        }});
    </script>
    <style>
        html, body {{
            background-color: #0B1220 !important;
            margin: 0;
            padding: 4px;
        }}
        .mermaid {{ display: flex; justify-content: center; }}
    </style>
    """
    components.html(html, height=height, scrolling=True)


def apply_chart_theme(fig: go.Figure, height: Optional[int] = None, title: Optional[str] = None) -> go.Figure:
    """
    Applies the institutional Bloomberg/Palantir dark theme to any Plotly figure.
    Sets dark navy transparent canvas, subtle gridlines, crisp monospaced ticks,
    and high-contrast typography.
    """
    layout_update = dict(
        paper_bgcolor="rgba(15, 23, 42, 0.7)",
        plot_bgcolor="rgba(7, 11, 20, 0.95)",
        font=dict(family="'Plus Jakarta Sans', sans-serif", color="#E2E8F0", size=12),
        margin=dict(l=50, r=30, t=55 if title else 40, b=45),
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="#C8A870",
            font=dict(family="'JetBrains Mono', monospace", color="#F8FAFC", size=12),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#1E293B",
            borderwidth=1,
            font=dict(color="#CBD5E1", size=11),
        ),
        xaxis=dict(
            gridcolor="rgba(30, 41, 59, 0.6)",
            linecolor="rgba(51, 65, 85, 0.6)",
            zerolinecolor="rgba(71, 85, 105, 0.4)",
            tickfont=dict(family="'JetBrains Mono', monospace", color="#94A3B8", size=11),
            title_font=dict(family="'Space Grotesk', sans-serif", color="#CBD5E1", size=13),
        ),
        yaxis=dict(
            gridcolor="rgba(30, 41, 59, 0.6)",
            linecolor="rgba(51, 65, 85, 0.6)",
            zerolinecolor="rgba(71, 85, 105, 0.4)",
            tickfont=dict(family="'JetBrains Mono', monospace", color="#94A3B8", size=11),
            title_font=dict(family="'Space Grotesk', sans-serif", color="#CBD5E1", size=13),
        ),
    )

    if height:
        layout_update["height"] = height
    if title:
        layout_update["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(family="'Space Grotesk', sans-serif", size=15, color="#F8FAFC"),
            x=0.02,
            y=0.96,
        )

    fig.update_layout(**layout_update)
    return fig
