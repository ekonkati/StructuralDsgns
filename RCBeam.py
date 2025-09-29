# Streamlit RCC Beam Designer (IS 456) – Mobile-friendly single file app
# --------------------------------------------------------------
# Removed matplotlib dependency to avoid ModuleNotFoundError.
# Uses Streamlit's built-in charting (line_chart) instead of matplotlib.
# --------------------------------------------------------------

import math
from dataclasses import dataclass
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RCC Beam Designer (IS 456)", layout="centered")
st.title("RCC Beam Designer – IS 456")
st.caption("Mobile-friendly Streamlit app. Units: length in m/mm, loads in kN.")

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
        return max(self.D - (self.cover + 8 + 0.5*16), 0.0)

# τ_bd table (design bond stress) – plain bars per IS456 Table 26; deformed +60%
TAU_BD_PLAIN = {20:1.2, 25:1.4, 30:1.5, 35:1.7, 40:1.9, 45:2.0, 50:2.2}

# τ_c(max) approximate per IS456 Table 20 (concrete shear strength max), N/mm2
TAU_C_MAX = {20:2.8, 25:3.1, 30:3.5, 35:3.7, 40:4.0}

# τ_c interpolation tables for p_t
TC_TABLE = {
    20: [(0.25,0.28),(0.50,0.38),(0.75,0.46),(1.00,0.62)],
    25: [(0.25,0.29),(0.50,0.40),(0.75,0.48),(1.00,0.62)],
    30: [(0.25,0.29),(0.50,0.40),(0.75,0.48),(1.00,0.62)],
    35: [(0.25,0.30),(0.50,0.42),(0.75,0.50),(1.00,0.62)],
    40: [(0.25,0.31),(0.50,0.44),(0.75,0.52),(1.00,0.62)],
}

LD_LIMITS = {"Simply Supported":20.0, "Continuous":26.0, "Cantilever":7.0}

def interp_xy(table, x):
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
    return interp_xy(TC_TABLE[fck_key], clamp(p_t, 0.25, 1.0))

def mu_lim_rect(fck, b_mm, d_mm, fy):
    k = 0.138 if fy <= 415 else 0.133
    MuNmm = k * fck * b_mm * (d_mm**2)
    return MuNmm / 1e6

def ast_singly(Mu_kNm, fy, d_mm, jd_ratio=0.9):
    Mu_Nmm = Mu_kNm * 1e6
    jd = jd_ratio * d_mm
    return Mu_Nmm / (0.87 * fy * jd)

def shear_design(Vu_kN, b_mm, d_mm, fck, fy, Ast_mm2):
    tau_v = (Vu_kN * 1e3) / (b_mm * d_mm)
    p_t = 100.0 * Ast_mm2 / (b_mm * d_mm)
    tc = tau_c(fck, p_t)
    tc_max = TAU_C_MAX[min(TAU_C_MAX.keys(), key=lambda k: abs(k - fck))]
    result = {"tau_v": tau_v, "p_t": p_t, "tau_c": tc, "tau_c_max": tc_max,
              "ok_concrete": tau_v <= tc, "exceeds_tcmax": tau_v > tc_max}
    if tau_v <= tc:
        result.update({"stirrups": "Provide minimum shear reinforcement", "s_v_mm": None})
    else:
        Vus_kN = Vu_kN - tc * b_mm * d_mm / 1e3
        phi, legs = 8.0, 2
        Asv = legs * math.pi * (phi**2) / 4.0
        s_v = (0.87 * fy * Asv * d_mm) / (Vus_kN * 1e3)
        result.update({"s_v_mm": s_v,
                       "stirrups": f"{int(phi)} mm {legs}-leg stirrups @ {max(50, min(300, round(s_v/10)*10))} mm c/c"})
    return result

def ld_required(fck, fy, bar_dia_mm, deformed=True, tension=True):
    fck_key = min(TAU_BD_PLAIN.keys(), key=lambda k: abs(k - fck))
    tau_bd = TAU_BD_PLAIN[fck_key]
    if deformed: tau_bd *= 1.6
    if not tension: tau_bd *= 1.25
    Ld = bar_dia_mm * fy / (4.0 * tau_bd)
    return Ld, tau_bd

# ---------- UI ----------
with st.sidebar:
    st.header("Inputs")
    span = st.number_input("Clear span L (m)", value=6.0, min_value=0.5, step=0.1)
    support = st.selectbox("End condition", ["Simply Supported","Continuous","Cantilever"], index=0)
    b = st.number_input("Beam width b (mm)", value=300, step=10)
    D = st.number_input("Overall depth D (mm)", value=500, step=10)
    cover = st.number_input("Clear cover (mm)", value=25, step=5)
    fck = st.selectbox("Concrete grade fck", [20,25,30,35,40], index=2)
    fy = st.selectbox("Steel grade fy", [415,500], index=1)
    finishes = st.number_input("Finishes (kN/m)", value=2.0, step=0.1)
    ll = st.number_input("Live load (kN/m)", value=5.0, step=0.5)
    t_bar = st.selectbox("Main bar dia (mm)", [12,16,20,25], index=1)
    c_bar = st.selectbox("Comp bar dia (mm)", [12,16,20,25], index=0)

mat, sec = Materials(fck,fy), Section(b,D,cover)
d = sec.d_eff
self_wt = mat.density * (b/1000.0) * (D/1000.0)

w_DL, w_LL = self_wt + finishes, ll
w_ULS_15 = 1.5*(w_DL+w_LL)

if support=="Simply Supported": kM,kV=1/8,0.5
elif support=="Cantilever": kM,kV=1/2,1.0
else: kM,kV=1/12,0.6

L=span
Mu_kNm=kM*w_ULS_15*(L**2)
Vu_kN=kV*w_ULS_15*L

Mu_lim=mu_lim_rect(mat.fck,sec.b,d,mat.fy)
need_double=Mu_kNm>Mu_lim
Ast_req=ast_singly(Mu_kNm,mat.fy,d)
area_one_bar=math.pi*(t_bar**2)/4.0
n_bars=max(2,math.ceil(Ast_req/area_one_bar))
Ast_prov=n_bars*area_one_bar

shear=shear_design(Vu_kN,sec.b,d,mat.fck,mat.fy,Ast_prov)
Ld_tension,tau_bd=ld_required(mat.fck,mat.fy,t_bar)
Ld_comp,_=ld_required(mat.fck,mat.fy,c_bar, tension=False)

base_Ld=LD_LIMITS[support]
p_t=100.0*Ast_prov/(sec.b*d)
mod=clamp(1.0+0.15*(p_t-1.0),0.8,1.3)
allowable_L_over_d=base_Ld*mod
actual_L_over_d=(L*1000.0)/d

# ---------- Output ----------
st.subheader("Summary")
st.metric("Mu",f"{Mu_kNm:.1f} kNm")
st.metric("Vu",f"{Vu_kN:.1f} kN")
st.write(f"Provide {n_bars} × {t_bar} mm bars (Ast={Ast_prov:.0f} mm²)")

st.subheader("Shear")
st.write(f"τv={shear['tau_v']:.3f}, τc={shear['tau_c']:.3f}, τcmax={shear['tau_c_max']:.2f}")
st.write(shear['stirrups'])

st.subheader("Development Length")
st.write(f"Ld tension={Ld_tension:.0f} mm, compression={Ld_comp:.0f} mm")

st.subheader("Serviceability")
st.write(f"L/d actual={actual_L_over_d:.1f}, allowable={allowable_L_over_d:.1f}")

# Use Streamlit chart instead of matplotlib
st.subheader("Moment & Shear Diagrams")
import numpy as np
xs=np.linspace(0,L,50)
M=[kM*w_ULS_15*(L*x-x*x) if support!="Cantilever" else -0.5*w_ULS_15*(x**2) for x in xs]
V=[w_ULS_15*(L/2-x) if support!="Cantilever" else -w_ULS_15*x for x in xs]

st.line_chart({"x":xs,"Moment (kNm)":M})
st.line_chart({"x":xs,"Shear (kN)":V})

# CSV download
summary={"span":[L],"support":[support],"Mu":[Mu_kNm],"Vu":[Vu_kN],
         "Ast_req":[Ast_req],"Ast_prov":[Ast_prov],"n_bars":[n_bars],
         "bar_dia":[t_bar],"Ld_tension":[Ld_tension],
         "allowable_L/d":[allowable_L_over_d],"actual_L/d":[actual_L_over_d]}
df=pd.DataFrame(summary)
buf=io.StringIO();df.to_csv(buf,index=False)
st.download_button("Download CSV",data=buf.getvalue(),file_name="beam_summary.csv")

st.caption("Educational tool per IS 456. Perform full detailing & checks separately.")
