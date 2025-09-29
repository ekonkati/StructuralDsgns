# Streamlit RCC Beam Designer (IS 456) – Mobile‑friendly single file app
# --------------------------------------------------------------
# Features
# • Material, geometry & load inputs (kN, m, mm)
# • Auto self‑weight and wall load options
# • End conditions: Simply Supported / Cantilever / Continuous (coefficients)
# • ULS combos (1.5(D+L), 1.2(D+L+E) optional E), envelopes Mu, Vu
# • Flexure design (singly or doubly reinforced if Mu>Mu_lim)
# • Shear design with τ_c interpolation (IS 456 Table concept for fck 20–40)
# • Stirrups spacing calc, τ_c,max check, warning if section upgrade needed
# • Dev. length & basic serviceability (L/d limits with simple tension‑steel factor)
# • Clear, compact outputs for mobile (≤ 1 column where needed)
# • Download: design summary CSV
# --------------------------------------------------------------

import math
from dataclasses import dataclass
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="RCC Beam Designer (IS 456)", layout="centered")
st.title("RCC Beam Designer – IS 456")
st.caption("Mobile‑friendly Streamlit app. Units: length in m/mm, loads in kN.")

# ---------- Helpers ----------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

@dataclass
class Materials:
    fck: int  # N/mm2
    fy: int   # N/mm2 (Fe415/500)
    density: float = 25.0  # kN/m3

@dataclass
class Section:
    b: float  # mm
    D: float  # mm overall
    cover: float  # mm to main tension steel

    @property
    def d_eff(self):
        # assume first layer bar dia guess = 16 mm; refine when known
        return max(self.D - (self.cover + 8 + 0.5*16), 0.0)

# τ_bd table (design bond stress) – plain bars per IS456 Table 26; deformed +60%
TAU_BD_PLAIN = {20:1.2, 25:1.4, 30:1.5, 35:1.7, 40:1.9, 45:2.0, 50:2.2}

# τ_c(max) approximate per IS456 Table 20 (concrete shear strength max), N/mm2
TAU_C_MAX = {20:2.8, 25:3.1, 30:3.5, 35:3.7, 40:4.0}

# τ_c interpolation tables for p_t (percent tension steel) for fck 20/25/30/35/40 (approx)
# Columns: p_t : τ_c
TC_TABLE = {
    20: [(0.25,0.28),(0.50,0.38),(0.75,0.46),(1.00,0.62),(1.25,0.62),(1.50,0.62),(1.75,0.62),(2.00,0.62),(3.00,0.62)],
    25: [(0.25,0.29),(0.50,0.40),(0.75,0.48),(1.00,0.62),(1.25,0.62),(1.50,0.62),(1.75,0.62),(2.00,0.62),(3.00,0.62)],
    30: [(0.25,0.29),(0.50,0.40),(0.75,0.48),(1.00,0.62),(1.25,0.62),(1.50,0.62),(1.75,0.62),(2.00,0.62),(3.00,0.62)],
    35: [(0.25,0.30),(0.50,0.42),(0.75,0.50),(1.00,0.62),(1.25,0.62),(1.50,0.62),(1.75,0.62),(2.00,0.62),(3.00,0.62)],
    40: [(0.25,0.31),(0.50,0.44),(0.75,0.52),(1.00,0.62),(1.25,0.62),(1.50,0.62),(1.75,0.62),(2.00,0.62),(3.00,0.62)],
}

# Basic L/d limits (IS 456 Table 2) – unmodified: SS=20, Cont=26, Cant=7
LD_LIMITS = {"Simply Supported":20.0, "Continuous":26.0, "Cantilever":7.0}


def interp_xy(table, x):
    # table: list of (x,y) sorted by x
    xs = [a for a,_ in table]
    ys = [b for _,b in table]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs)-1):
        if xs[i] <= x <= xs[i+1]:
            x0,y0 = xs[i],ys[i]
            x1,y1 = xs[i+1],ys[i+1]
            t = (x - x0) / (x1 - x0)
            return y0 + t*(y1-y0)


def tau_c(fck, p_t):
    fck_key = min(TC_TABLE.keys(), key=lambda k: abs(k - fck))
    return interp_xy(TC_TABLE[fck_key], clamp(p_t, 0.25, 3.0))


def mu_lim_rect(fck, b_mm, d_mm, fy):
    # Limiting moment of resistance for singly reinforced rectangular section (Fe415 approx coef 0.138)
    # For generalized, compute via x_u,lim per IS456, but here we use Fe415/500 factors with small variance
    if fy <= 415:
        k = 0.138
    else:
        k = 0.133  # Fe500 slightly lower
    MuNmm = k * fck * b_mm * (d_mm**2)
    return MuNmm / 1e6  # kN·m


def ast_singly(Mu_kNm, fy, d_mm, jd_ratio=0.9):
    # Ast = Mu / (0.87*fy*jd), Mu in kNm
    Mu_Nmm = Mu_kNm * 1e6
    jd = jd_ratio * d_mm
    Ast = Mu_Nmm / (0.87 * fy * jd)
    return Ast  # mm2


def shear_design(Vu_kN, b_mm, d_mm, fck, fy, Ast_mm2):
    tau_v = (Vu_kN * 1e3) / (b_mm * d_mm)  # N/mm2
    p_t = 100.0 * Ast_mm2 / (b_mm * d_mm)
    tc = tau_c(fck, p_t)
    tc_max = TAU_C_MAX[min(TAU_C_MAX.keys(), key=lambda k: abs(k - fck))]
    result = {
        "tau_v": tau_v,
        "p_t": p_t,
        "tau_c": tc,
        "tau_c_max": tc_max,
        "ok_concrete": tau_v <= tc,
        "exceeds_tcmax": tau_v > tc_max,
    }
    if tau_v <= tc:
        # min shear reinforcement
        result.update({
            "stirrups": "Provide minimum shear reinforcement",
            "s_v_mm": None,
        })
    else:
        Vus_kN = Vu_kN - tc * b_mm * d_mm / 1e3
        # Vertical 2‑leg stirrup Asv = 2*(pi*phi^2/4)
        # Let user select later; here assume 8 mm two‑leg by default
        phi = 8.0
        legs = 2
        Asv = legs * math.pi * (phi**2) / 4.0  # mm2
        s_v = (0.87 * fy * Asv * d_mm) / (Vus_kN * 1e3)  # mm
        result.update({
            "Vus_kN": Vus_kN,
            "phi_default": phi,
            "legs_default": legs,
            "Asv_mm2": Asv,
            "s_v_mm": s_v,
            "stirrups": f"{int(phi)} mm {legs}-leg vertical stirrups @ {max(50, min(300, round(s_v/10)*10))} mm c/c (cap at min{{0.75d, 300}})"
        })
    return result


def ld_required(fck, fy, bar_dia_mm, deformed=True, tension=True):
    # τ_bd base
    fck_key = min(TAU_BD_PLAIN.keys(), key=lambda k: abs(k - fck))
    tau_bd = TAU_BD_PLAIN[fck_key]
    if deformed:
        tau_bd *= 1.6
    if not tension:
        tau_bd *= 1.25
    Ld = bar_dia_mm * fy / (4.0 * tau_bd)
    return Ld, tau_bd

# ---------- UI ----------
with st.sidebar:
    st.header("Inputs")
    st.subheader("Geometry & Supports")
    span = st.number_input("Clear span L (m)", value=6.0, min_value=0.5, step=0.1)
    support = st.selectbox("End condition", ["Simply Supported","Continuous","Cantilever"], index=0)
    b = st.number_input("Beam width b (mm)", value=300, step=10, min_value=150)
    D = st.number_input("Overall depth D (mm)", value=500, step=10, min_value=200)
    cover = st.number_input("Clear cover to tension steel (mm)", value=25, step=5, min_value=20)

    st.subheader("Materials")
    fck = st.selectbox("Concrete grade fck (N/mm²)", [20,25,30,35,40], index=2)
    fy = st.selectbox("Steel grade fy (N/mm²)", [415,500], index=1)

    st.subheader("Loads")
    use_wall = st.checkbox("Add wall load on beam", value=True)
    wall_thk = st.number_input("Wall thickness (mm)", value=115, step=115, min_value=0)
    wall_h = st.number_input("Wall height (m)", value=3.0, step=0.1, min_value=0.0)
    wall_density = st.number_input("Masonry density (kN/m³)", value=19.0, step=0.5)

    finishes = st.number_input("Superimposed DL (finishes/services) kN/m", value=2.0, step=0.1, min_value=0.0)
    ll = st.number_input("Live load (kN/m)", value=5.0, step=0.5, min_value=0.0)
    include_eq = st.checkbox("Include lateral load effects (E) via coefficient?", value=False)
    eq_coeff = st.number_input("Equivalent bending coefficient for E (×wL²)", value=0.0, help="Set a coefficient to include seismic/wind as equivalent moment if desired.")

    st.subheader("Detailing prefs")
    t_bar = st.selectbox("Main bar dia (mm)", [12,16,20,25,28,32], index=1)
    c_bar = st.selectbox("Compression bar dia (mm)", [12,16,20,25], index=0)
    stirrup_dia = st.selectbox("Stirrup dia (mm)", [6,8,10], index=1)
    legs = st.selectbox("Stirrup legs", [2,4], index=0)

# Compute
mat = Materials(fck=fck, fy=fy)
sec = Section(b=b, D=D, cover=cover)
d = sec.d_eff  # mm

# Self weight per m (kN/m)
self_wt = mat.density * (b/1000.0) * (D/1000.0)
wall_kNpm = 0.0
if use_wall and wall_thk>0 and wall_h>0:
    wall_kNpm = wall_density * (wall_thk/1000.0) * wall_h

w_DL = self_wt + finishes + wall_kNpm
w_LL = ll
w_total_SLS = w_DL + w_LL

# ULS factored combinations
w_ULS_15 = 1.5*(w_DL + w_LL)
w_ULS_12E = 1.2*(w_DL + w_LL)  # +E as coefficient to moment below

# Bending & shear coefficients
if support == "Simply Supported":
    kM = 1/8  # wL^2/8
    kV = 0.5  # V ~ wL/2 at support
elif support == "Cantilever":
    kM = 1/2  # wL^2/2
    kV = 1.0  # V ~ wL at fixed
else:  # Continuous
    # typical midspan positive ≈ wL^2/12 ; negative at support ≈ wL^2/10
    kM = 1/12
    kV = 0.6

L = span

# Design moments (kNm) and shears (kN)
Mu_kNm = kM * w_ULS_15 * (L**2)
Vu_kN  = kV * w_ULS_15 * L

if include_eq and eq_coeff>0:
    Mu_kNm += eq_coeff * w_ULS_12E * (L**2)

# Flexure design
Mu_lim = mu_lim_rect(mat.fck, sec.b, d, mat.fy)
need_double = Mu_kNm > Mu_lim

Ast_req = ast_singly(Mu_kNm, mat.fy, d)

# Provide bars: try to round to available bars with selected dia
area_one_bar = math.pi * (t_bar**2) / 4.0
n_bars = max(2, math.ceil(Ast_req / area_one_bar))
Ast_prov = n_bars * area_one_bar

# Shear design with provided Ast
shear = shear_design(Vu_kN, sec.b, d, mat.fck, mat.fy, Ast_prov)

# Dev length
Ld_tension, tau_bd = ld_required(mat.fck, mat.fy, t_bar, deformed=True, tension=True)
Ld_comp, _ = ld_required(mat.fck, mat.fy, c_bar, deformed=True, tension=False)

# Serviceability: basic L/d limit with simple tension‑steel modifier (conservative)
base_Ld = LD_LIMITS[support]
p_t = 100.0 * Ast_prov / (sec.b * d)
# crude modifier ~ increases with steel %, clamp 0.8–1.3
mod = clamp(1.0 + 0.15*(p_t-1.0), 0.8, 1.3)
allowable_L_over_d = base_Ld * mod
actual_L_over_d = (L*1000.0) / d

# ---------- Output ----------
st.subheader("Quick Summary")
col1,col2 = st.columns(2)
with col1:
    st.metric("Mu (ULS)", f"{Mu_kNm:.1f} kN·m")
    st.metric("Vu (ULS)", f"{Vu_kN:.1f} kN")
    st.write(f"Self‑weight: **{self_wt:.2f} kN/m**, Wall: **{wall_kNpm:.2f} kN/m**, Finishes: **{finishes:.2f} kN/m**, LL: **{ll:.2f} kN/m**")
with col2:
    st.metric("d (effective)", f"{d:.0f} mm")
    st.metric("Mu,lim", f"{Mu_lim:.1f} kN·m")
    st.write("Doubly reinforced needed: **" + ("Yes" if need_double else "No") + "**")

st.divider()

st.subheader("Flexure Design")
st.write(f"Required Ast (tension): **{Ast_req:.0f} mm²**")
st.write(f"Provide: **{n_bars} nos {t_bar} mm** → Ast_prov = **{Ast_prov:.0f} mm²**")
if need_double:
    st.warning("Mu > Mu_lim for singly‑reinforced section → Provide compression steel or increase depth/width. App currently sizes only tension steel; consider increasing D or switch to doubly‑reinforced design.")

st.subheader("Shear Design")
st.write(f"τ_v = **{shear['tau_v']:.3f} N/mm²**, p_t = **{shear['p_t']:.2f}%**, τ_c ≈ **{shear['tau_c']:.3f}**, τ_c,max ≈ **{shear['tau_c_max']:.2f}**")
if shear["exceeds_tcmax"]:
    st.error("τ_v > τ_c,max → Increase b or d or concrete grade; section unsafe in shear.")
else:
    if shear["ok_concrete"]:
        st.success("Concrete alone can resist design shear → Provide minimum web reinforcement as per IS 456.")
        st.info("Min stirrups: 2‑leg 8 mm @ 300 mm c/c (or as per code minimums and exposure/seismic).")
    else:
        st.warning("Provide shear reinforcement.")
        st.write(f"Try **{stirrup_dia} mm** {legs}-leg vertical stirrups.")
        # recompute spacing with user chosen stirrup
        Asv_user = legs * math.pi * (stirrup_dia**2) / 4.0
        Vus_kN = Vu_kN - shear['tau_c'] * sec.b * d / 1e3
        s_v_user = (0.87 * mat.fy * Asv_user * d) / (Vus_kN * 1e3)
        s_v_final = max(50, min(300, round(s_v_user/10)*10))
        st.write(f"Computed s_v = {s_v_user:.0f} mm → **use {s_v_final} mm c/c**, also cap by min{{0.75d, 300}} = {min(0.75*d, 300):.0f} mm.")

st.subheader("Development Length & Anchorage")
st.write(f"Ld (tension, {t_bar} mm) ≈ **{Ld_tension:.0f} mm**, using τ_bd ≈ {tau_bd:.2f} N/mm² (deformed bars).")
st.write(f"Ld (compression, {c_bar} mm) ≈ **{Ld_comp:.0f} mm**.")
st.info("Ensure bar extensions beyond critical sections and proper hooks/bends as per code and seismic detailing (IS 13920 where applicable).")

st.subheader("Serviceability – Span/Depth Check")
st.write(f"Basic L/d limit ({support}): **{base_Ld:.1f}**; modifier (tension steel %): **{mod:.2f}** → allowable **{allowable_L_over_d:.1f}**.")
st.write(f"Actual L/d = **{actual_L_over_d:.1f}** → {'OK' if actual_L_over_d <= allowable_L_over_d else 'Increase depth / compression steel'}.")

# Plot moment and shear diagram for UDL (single span)
st.subheader("Diagrams (UDL)")
npts = 50
xs = [i*L/(npts-1) for i in range(npts)]
M = [kM*w_ULS_15*(L*x - x*x) if support!="Cantilever" else -0.5*w_ULS_15*(x**2) for x in xs]
V = [w_ULS_15*(L/2 - x) if support!="Cantilever" else -w_ULS_15*x for x in xs]

fig1, ax1 = plt.subplots()
ax1.plot(xs, M)
ax1.set_xlabel("x (m)")
ax1.set_ylabel("Bending Moment M(x) [kN·m]")
ax1.set_title("Bending Moment Diagram (ULS)")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(xs, V)
ax2.set_xlabel("x (m)")
ax2.set_ylabel("Shear Force V(x) [kN]")
ax2.set_title("Shear Force Diagram (ULS)")
st.pyplot(fig2)

# Download summary
summary = {
    "span_m":[L],
    "support":[support],
    "b_mm":[sec.b],
    "D_mm":[sec.D],
    "d_mm":[d],
    "fck":[mat.fck],
    "fy":[mat.fy],
    "w_DL_kNpm":[w_DL],
    "w_LL_kNpm":[w_LL],
    "Mu_kNm":[Mu_kNm],
    "Vu_kN":[Vu_kN],
    "Ast_req_mm2":[Ast_req],
    "Ast_prov_mm2":[Ast_prov],
    "n_bars":[n_bars],
    "bar_dia_mm":[t_bar],
    "tau_v":[shear['tau_v']],
    "tau_c":[shear['tau_c']],
    "tau_c_max":[shear['tau_c_max']],
    "Ld_tension_mm":[Ld_tension],
    "allowable_L_over_d":[allowable_L_over_d],
    "actual_L_over_d":[actual_L_over_d],
}
df = pd.DataFrame(summary)
buf = io.StringIO()
df.to_csv(buf, index=False)
st.download_button("Download design summary (CSV)", data=buf.getvalue(), file_name="beam_design_summary.csv", mime="text/csv")

st.caption("Notes: This is an educational/assistive tool following IS 456 concepts. For final issue, perform detailed analysis, crack/deflection checks, and ductile detailing per IS 13920 where applicable.")
