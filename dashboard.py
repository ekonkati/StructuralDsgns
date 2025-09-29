# Structural Design Dashboard — Streamlit
# One hub to host multiple component designers you'll add over time.
# Provides: sidebar navigation, plugin registry, project save/load, unit & code settings,
# simple example pages (RCC Column, RCC Beam, Footing, Slab, Stair, Lintel, Canopy).
#
# How to extend:
# - Drop new modules under a `components/` folder as Python files exporting a `register()`
#   function that returns dict entries {"name":..., "key":..., "icon":..., "run":callable}.
# - Or add stubs in `BUILT_INS` below and flesh them out later.

import json
import math
import os
import pkgutil
import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple

import numpy as np
import streamlit as st

# ---------------------------------
# Global Config & Theming
# ---------------------------------
st.set_page_config(
    page_title="Structural Design Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------
# Utilities & Shared Helpers
# ---------------------------------
ES = 200000.0  # MPa
EPS_CU = 0.0035


def Ec_from_fck(fck: float) -> float:
    return 5000.0 * math.sqrt(max(fck, 1e-6))


def bar_area(d_mm: float) -> float:
    return math.pi * (d_mm**2) / 4.0


@dataclass
class Section:
    b: float
    D: float
    cover: float
    bars: List[Tuple[float, float, float]]  # (x, y, dia)

    @property
    def Ag(self) -> float:
        return self.b * self.D

    @property
    def Ic_x(self) -> float:
        return self.b * self.D**3 / 12.0

    @property
    def Ic_y(self) -> float:
        return self.D * self.b**3 / 12.0

    @property
    def rx(self) -> float:
        return math.sqrt(self.Ic_x / self.Ag)

    @property
    def ry(self) -> float:
        return math.sqrt(self.Ic_y / self.Ag)


def generate_rectangular_bar_layout(b: float, D: float, cover: float,
                                    n_top: int, n_bot: int, n_left: int, n_right: int,
                                    dia_top: float, dia_bot: float, dia_side: float) -> List[Tuple[float, float, float]]:
    bars = []
    def linspace(a, c, n):
        if n <= 1:
            return [0.5 * (a + c)]
        return [a + i * (c - a) / (n - 1) for i in range(n)]
    # top row
    y_top = cover + dia_top/2
    xL = cover + dia_side/2
    xR = b - (cover + dia_side/2)
    for x in linspace(xL, xR, n_top):
        bars.append((x, y_top, dia_top))
    # bottom row
    y_bot = D - (cover + dia_bot/2)
    for x in linspace(xL, xR, n_bot):
        bars.append((x, y_bot, dia_bot))
    # sides
    x_left = xL
    x_right = xR
    yT = y_top
    yB = y_bot
    if n_left > 2:
        for y in linspace(yT, yB, n_left)[1:-1]:
            bars.append((x_left, y, dia_side))
    if n_right > 2:
        for y in linspace(yT, yB, n_right)[1:-1]:
            bars.append((x_right, y, dia_side))
    return bars


def svg_cross_section(section: Section, tie_dia: float, tie_spacing: float) -> str:
    b, D = section.b, section.D
    scale = 1.0
    pad = 40
    W = int(b*scale + 2*pad)
    H = int(D*scale + 2*pad)
    def px(mm):
        return mm*scale + pad
    parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
             '<style>.txt{font:12px sans-serif}.lbl{font:11px sans-serif}</style>']
    parts.append(f'<rect x="{px(0)}" y="{px(0)}" width="{b}" height="{D}" fill="#f2f2f2" stroke="#444"/>')
    cov = section.cover
    parts.append(f'<rect x="{px(cov)}" y="{px(cov)}" width="{b-2*cov}" height="{D-2*cov}" fill="none" stroke="#888" stroke-dasharray="4,3"/>')
    for i,(x,y,d) in enumerate(section.bars,1):
        cx, cy, r = px(x), px(y), d/2
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#0a66c2" stroke="#043f77"/>')
        lx, ly = cx+30, cy-10
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{lx}" y2="{ly}" stroke="#333"/>')
        parts.append(f'<text x="{lx+4}" y="{ly-4}" class="lbl">Bar {i}: {int(d)}mm</text>')
    parts.append(f'<text x="{px(b/2)}" y="{px(-10)}" class="txt" text-anchor="middle">b={b:.0f}mm</text>')
    parts.append(f'<text x="{px(-10)}" y="{px(D/2)}" class="txt" text-anchor="end" transform="rotate(-90 {px(-10)} {px(D/2)})">D={D:.0f}mm</text>')
    parts.append(f'<text x="{px(b+5)}" y="{px(15)}" class="lbl">Tie {int(tie_dia)} @ {int(tie_spacing)} c/c</text>')
    parts.append('</svg>')
    return "\n".join(parts)


# ---------------------------------
# Project Model (save/load JSON)
# ---------------------------------
DEFAULT_SETTINGS = {
    "units": "SI",
    "code_set": "IS (456, 13920, 875, 1893)",
    "project_name": "Untitled Project",
    "location": "",
}

if "project" not in st.session_state:
    st.session_state.project = {
        "settings": DEFAULT_SETTINGS.copy(),
        "data": {},  # each component can write its data here under its key
    }

PROJECT = st.session_state.project


def save_project_button():
    payload = json.dumps(PROJECT, indent=2)
    st.download_button("💾 Download Project JSON", data=payload, file_name="project.json", mime="application/json")


def load_project_uploader():
    up = st.file_uploader("Upload Project JSON", type=["json"], accept_multiple_files=False)
    if up is not None:
        try:
            loaded = json.load(up)
            st.session_state.project = loaded
            st.success("Project loaded.")
        except Exception as e:
            st.error(f"Failed to load: {e}")


# ---------------------------------
# Plugin Registry
# ---------------------------------
Component = Dict[str, Any]  # keys: name, key, icon, run(state)
REGISTRY: Dict[str, Component] = {}


def register_component(comp: Component):
    REGISTRY[comp["key"]] = comp


# Built-in stub pages (you can replace with real ones later)

def page_welcome(state):
    st.title("🏗️ Structural Design Dashboard")
    st.markdown("Use the sidebar to open a component. Save/Load projects from the top bar.")
    st.info("Tip: Create a `components/` folder with modules exporting `register()` to auto-appear here.")


def page_rcc_column(state):
    st.header("RCC Column — quick stub (biaxial)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        b = st.number_input("b (mm)", 200.0, 2000.0, 450.0, 25.0)
        fck = st.number_input("fck (MPa)", 20.0, 80.0, 30.0, 1.0)
    with c2:
        D = st.number_input("D (mm)", 200.0, 3000.0, 600.0, 25.0)
        fy = st.number_input("fy (MPa)", 415.0, 600.0, 500.0, 5.0)
    with c3:
        cover = st.number_input("Cover (mm)", 20.0, 75.0, 40.0, 5.0)
        tie_dia = st.selectbox("Tie dia", [6,8,10], index=1)
    with c4:
        s = st.number_input("Tie spacing (mm)", 50.0, 300.0, 150.0, 5.0)

    st.subheader("Bars")
    cL1, cL2, cL3, cL4 = st.columns(4)
    with cL1:
        n_top = st.number_input("Top bars", 0, 10, 3)
        dia_top = st.selectbox("Top dia", [12,16,20,25,28,32], index=1)
    with cL2:
        n_bot = st.number_input("Bottom bars", 0, 10, 3)
        dia_bot = st.selectbox("Bottom dia", [12,16,20,25,28,32], index=1)
    with cL3:
        n_left = st.number_input("Left bars", 0, 10, 2)
        dia_side = st.selectbox("Side dia", [12,16,20,25,28], index=0)
    with cL4:
        n_right = st.number_input("Right bars", 0, 10, 2)

    bars = generate_rectangular_bar_layout(b, D, cover, n_top, n_bot, n_left, n_right, float(dia_top), float(dia_bot), float(dia_side))
    section = Section(b, D, cover, bars)
    st.markdown("#### Cross-section preview")
    st.components.v1.html(svg_cross_section(section, tie_dia=float(tie_dia), tie_spacing=float(s)), height=min(600, int(D+100)))

    # Very light utilization placeholder
    st.markdown("---")
    st.markdown("#### Quick capacity placeholders (replace with your real checks)")
    Pu = st.number_input("Pu (kN, +comp)", -3000.0, 6000.0, 1200.0, 10.0) * 1e3
    Mux = st.number_input("Mux (kNm)", -2000.0, 2000.0, 120.0, 5.0) * 1e6
    Muy = st.number_input("Muy (kNm)", -2000.0, 2000.0, 80.0, 5.0) * 1e6

    # super-simplified dummy limits just for UI demonstration
    As_long = sum(bar_area(d) for _,_,d in bars)
    cap_dummy = 0.9 * fck * section.Ag + 0.87 * fy * As_long
    util_axial = Pu / max(cap_dummy, 1e-6)
    st.metric("Axial util (dummy)", f"{util_axial:.2f}")

    # persist minimal state
    state.setdefault("rcc_column", {})
    state["rcc_column"].update(dict(b=b, D=D, cover=cover, bars=bars, Pu=Pu, Mux=Mux, Muy=Muy))


def page_stub(name: str):
    def _run(state):
        st.header(f"{name} — placeholder")
        st.info("This is a stub. Plug your design logic here or add a module under `components/`.")
    return _run


BUILT_INS: List[Component] = [
    {"name": "Welcome", "key": "welcome", "icon": "🏠", "run": page_welcome},
    {"name": "RCC Column", "key": "rcc_column", "icon": "🧱", "run": page_rcc_column},
    {"name": "RCC Beam", "key": "rcc_beam", "icon": "📏", "run": page_stub("RCC Beam")},
    {"name": "Footing", "key": "footing", "icon": "🦶", "run": page_stub("Footing")},
    {"name": "Slab", "key": "slab", "icon": "🧊", "run": page_stub("Slab")},
    {"name": "Staircase", "key": "stair", "icon": "🪜", "run": page_stub("Staircase")},
    {"name": "Lintel", "key": "lintel", "icon": "🪟", "run": page_stub("Lintel")},
    {"name": "Canopy", "key": "canopy", "icon": "⛱️", "run": page_stub("Canopy")},
]


def load_plugins_from_folder(folder: str = "components"):
    if not os.path.isdir(folder):
        return
    for _, modname, ispkg in pkgutil.iter_modules([folder]):
        if ispkg:
            continue
        try:
            module = importlib.import_module(f"{folder}.{modname}")
            if hasattr(module, "register"):
                comps = module.register()
                if isinstance(comps, dict):
                    comps = [comps]
                for c in comps:
                    register_component(c)
        except Exception as e:
            st.sidebar.warning(f"Failed to load plugin {modname}: {e}")


# Register built-ins and external plugins
for c in BUILT_INS:
    register_component(c)
load_plugins_from_folder()

# ---------------------------------
# Top Bar (Settings / Save–Load)
# ---------------------------------
st.markdown(
    """
    <style>
    .topbar {display:flex; gap:8px; align-items:center;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    tb1, tb2, tb3, tb4 = st.columns([3,2,2,2])
    with tb1:
        st.text_input("Project name", key="proj_name", value=PROJECT["settings"].get("project_name", "Untitled Project"))
    with tb2:
        PROJECT["settings"]["units"] = st.selectbox("Units", ["SI", "Imperial"], index=0)
    with tb3:
        PROJECT["settings"]["code_set"] = st.selectbox("Code set", ["IS (456, 13920, 875, 1893)", "Eurocode", "ACI"], index=0)
    with tb4:
        save_project_button()
        load_project_uploader()

PROJECT["settings"]["project_name"] = st.session_state.get("proj_name", "Untitled Project")

# ---------------------------------
# Sidebar Navigation
# ---------------------------------
st.sidebar.title("Components")
search = st.sidebar.text_input("Search component")

# order by name, keep welcome first
keys_sorted = ["welcome"] + sorted([k for k in REGISTRY.keys() if k != "welcome"], key=lambda k: REGISTRY[k]["name"].lower())

filtered = [k for k in keys_sorted if search.strip().lower() in REGISTRY[k]["name"].lower()]
labels = [f"{REGISTRY[k]['icon']}  {REGISTRY[k]['name']}" for k in filtered]
sel = st.sidebar.radio("", options=filtered, format_func=lambda k: f"{REGISTRY[k]['icon']}  {REGISTRY[k]['name']}")

st.sidebar.markdown("---")
st.sidebar.subheader("How to add modules")
st.sidebar.caption(
    "Create `components/my_beam.py` with a `register()` that returns a dict: \n"
    "`{""name"": ""My Beam"", ""key"": ""my_beam"", ""icon"": ""📐"", ""run"": callable}`.\n"
    "Your callable receives `state` dict and can read/write `st.session_state.project['data'][key]`.")

# Ensure state bucket for component data exists
if "data" not in PROJECT:
    PROJECT["data"] = {}

# Provide a shared state dict to pages
STATE = PROJECT["data"]

# ---------------------------------
# Render Selected Page
# ---------------------------------
REGISTRY[sel]["run"](STATE)

# Footer
st.markdown("---")
st.caption("Dashboard scaffold • Add/plug your detailed designers incrementally • Exports via top bar")
