"""Gas Dispersion Dashboard - Gaussian Plume / Puff / Finite-duration release.
Run:  streamlit run app.py
"""
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go

# Allow running with PyCharm's Run button / "python app.py": relaunch through Streamlit
import sys, subprocess
from streamlit.runtime.scriptrunner import get_script_run_ctx
if get_script_run_ctx() is None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    sys.exit()

st.set_page_config(page_title="Dispersion Dashboard", page_icon="☁️", layout="wide")

# ---------------- Core models ----------------
def sigmas(x, cls, terrain):
    """Briggs sigma_y, sigma_z (m) for downwind distance x (m)."""
    x = np.maximum(x, 1.0)
    if terrain == "Rural":
        a = {"A": (.22, 0), "B": (.16, 0), "C": (.11, 0), "D": (.08, 0), "E": (.06, 0), "F": (.04, 0)}[cls][0]
        sy = a * x / np.sqrt(1 + 1e-4 * x)
        sz = {"A": .20 * x, "B": .12 * x, "C": .08 * x / np.sqrt(1 + 2e-4 * x),
              "D": .06 * x / np.sqrt(1 + 1.5e-3 * x), "E": .03 * x / (1 + 3e-4 * x),
              "F": .016 * x / (1 + 3e-4 * x)}[cls]
    else:
        k = {"A": "AB", "B": "AB", "C": "C", "D": "D", "E": "EF", "F": "EF"}[cls]
        a = {"AB": .32, "C": .22, "D": .16, "EF": .11}[k]
        sy = a * x / np.sqrt(1 + 4e-4 * x)
        sz = {"AB": .24 * x * np.sqrt(1 + 1e-3 * x), "C": .20 * x,
              "D": .14 * x / np.sqrt(1 + 3e-4 * x), "EF": .08 * x / np.sqrt(1 + 1.5e-3 * x)}[k]
    return sy, sz

def vert(z, H, sz):
    return np.exp(-(z - H) ** 2 / (2 * sz ** 2)) + np.exp(-(z + H) ** 2 / (2 * sz ** 2))

def puff_c(xd, yc, z, age, m, u, H, cls, ter):
    """Instantaneous release of m grams, at 'age' seconds after release."""
    xc = u * age
    sy, sz = sigmas(np.array(xc), cls, ter)
    return (m / ((2 * np.pi) ** 1.5 * sy * sy * sz) * np.exp(-(xd - xc) ** 2 / (2 * sy ** 2))
            * np.exp(-yc ** 2 / (2 * sy ** 2)) * vert(z, H, sz))

def conc(model, xd, yc, z, t, P):
    u, H, cls, ter, Q = P["u"], P["H"], P["cls"], P["ter"], P["Q"]
    if model == "Gaussian Plume (continuous, steady-state)":
        sy, sz = sigmas(xd, cls, ter)
        c = Q / (2 * np.pi * u * sy * sz) * np.exp(-yc ** 2 / (2 * sy ** 2)) * vert(z, H, sz)
        return np.where(xd > 0, c, 0.0)
    if model == "Gaussian Puff (instantaneous release)":
        return puff_c(xd, yc, z, max(t, 1.0), Q, u, H, cls, ter)
    # Finite-duration release: superposition of puffs (Q g/s for T seconds)
    T = P["T"]; te = min(t, T); n = 80
    s = (np.arange(n) + 0.5) * te / n
    dm = Q * te / n
    return sum(puff_c(xd, yc, z, max(t - si, 1.0), dm, u, H, cls, ter) for si in s)

def to_xy(E, N, wind_from):
    phi = np.radians((wind_from + 180) % 360)        # direction plume travels (compass)
    dE, dN = np.sin(phi), np.cos(phi)
    return E * dE + N * dN, -E * dN + N * dE, (dE, dN)

# ---------------- Sidebar inputs ----------------
st.sidebar.title("⚙️ Inputs")
model = st.sidebar.selectbox("Dispersion model", [
    "Gaussian Plume (continuous, steady-state)",
    "Gaussian Puff (instantaneous release)",
    "Finite-duration release (puff superposition)"])
unit_q = "g/s" if "Puff (inst" not in model else "g"
Q = st.sidebar.number_input(f"Release {'rate' if unit_q=='g/s' else 'amount'} ({unit_q})", 0.01, 1e8, 100.0 if unit_q == "g/s" else 50000.0)
T = 600.0
if model.startswith("Finite"):
    T = st.sidebar.number_input("Release duration (s)", 1.0, 1e5, 600.0)
u = st.sidebar.slider("Wind speed (m/s)", 0.5, 20.0, 3.0, 0.5)
wd = st.sidebar.slider("Wind FROM direction (° from North)", 0, 359, 270)
cls = st.sidebar.selectbox("Pasquill stability class", list("ABCDEF"), index=3,
                           help="A very unstable (sunny, light wind) ... D neutral ... F very stable (clear night)")
ter = st.sidebar.radio("Terrain", ["Rural", "Urban"], horizontal=True)
H = st.sidebar.number_input("Effective release height (m)", 0.0, 500.0, 5.0)
z = st.sidebar.number_input("Receptor / plot height z (m)", 0.0, 100.0, 1.5)
t = 600.0
if "Plume" not in model:
    t = st.sidebar.slider("Time since release (s)", 10, 7200, 600, 10)
st.sidebar.markdown("---")
thr = st.sidebar.number_input("Threshold concentration (mg/m³)", 0.0001, 1e6, 50.0, format="%.4f",
                              help="e.g. IDLH / ERPG / AEGL value of the chemical")
mw = st.sidebar.number_input("Molecular weight (g/mol, for ppm)", 1.0, 500.0, 17.03)
ext = st.sidebar.slider("Plot half-extent (m)", 100, 10000, 2000, 100)
rE = st.sidebar.number_input("Receptor East (m)", -20000.0, 20000.0, -500.0)
rN = st.sidebar.number_input("Receptor North (m)", -20000.0, 20000.0, 0.0)

P = dict(u=u, H=H, cls=cls, ter=ter, Q=Q, T=T)

# ---------------- Compute ----------------
n = 260
E, N = np.meshgrid(np.linspace(-ext, ext, n), np.linspace(-ext, ext, n))
xd, yc, (dE, dN) = to_xy(E, N, wd)
C = conc(model, xd, yc, z, t, P) * 1e3                     # mg/m3
cell = (2 * ext / (n - 1)) ** 2
mask = C >= thr
area_ha = mask.sum() * cell / 1e4
reach = np.hypot(E[mask], N[mask]).max() if mask.any() else 0.0
rx, ry, _ = to_xy(rE, rN, wd)
c_rec = float(conc(model, rx, ry, z, t, P)) * 1e3

# ---------------- Dashboard ----------------
st.title("☁️ Gas Dispersion & Affected-Zone Dashboard")
st.caption(f"{model} | class {cls}, {ter.lower()} | u = {u} m/s | wind from {wd}°")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Max concentration (mg/m³)", f"{C.max():.3g}")
m2.metric("Receptor concentration", f"{c_rec:.3g} mg/m³", f"{c_rec*24.45/mw:.3g} ppm", delta_color="off")
m3.metric("Affected area (> threshold)", f"{area_ha:.2f} ha")
m4.metric("Max hazard distance", f"{reach:.0f} m")
if c_rec >= thr:
    st.error("⚠️ Receptor is INSIDE the affected zone (above threshold).")
else:
    st.success("✅ Receptor is below the threshold at this time.")

tab1, tab2, tab3 = st.tabs(["🗺️ Contour map", "📈 Profiles", "📘 Model notes"])

with tab1:
    logC = np.log10(np.maximum(C, 1e-12))
    top = np.log10(max(C.max(), thr * 1.01, 1e-9))
    fig = go.Figure(go.Contour(
        x=np.linspace(-ext, ext, n), y=np.linspace(-ext, ext, n), z=logC,
        colorscale="Turbo", zmin=top - 4, zmax=top, ncontours=16,
        contours=dict(showlines=False),
        colorbar=dict(title="log10 C (mg/m³)"),
        hovertemplate="E %{x:.0f} m<br>N %{y:.0f} m<br>log10 C %{z:.2f}<extra></extra>"))
    fig.add_trace(go.Contour(x=np.linspace(-ext, ext, n), y=np.linspace(-ext, ext, n), z=logC,
        contours=dict(type="constraint", operation=">=", value=np.log10(thr), coloring="lines"),
        line=dict(color="white", width=3), showscale=False, name="Threshold",
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=["Source"], textposition="top center",
                             marker=dict(symbol="triangle-up", size=14, color="black"), name="Source"))
    fig.add_trace(go.Scatter(x=[rE], y=[rN], mode="markers+text", text=["Receptor"], textposition="bottom center",
                             marker=dict(symbol="star", size=14, color="white", line=dict(color="black", width=1)), name="Receptor"))
    fig.add_annotation(x=dE * ext * .35, y=dN * ext * .35, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor="black", text="Wind →")
    fig.update_layout(height=700, xaxis_title="East (m)", yaxis=dict(title="North (m)", scaleanchor="x"),
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("White outline = affected zone (concentration ≥ threshold).")
    df = pd.DataFrame({"East_m": E.ravel(), "North_m": N.ravel(), "Conc_mg_m3": C.ravel()})
    st.download_button("⬇️ Download concentration grid (CSV)", df.to_csv(index=False), "concentration_grid.csv")

with tab2:
    c1, c2 = st.columns(2)
    xs = np.linspace(1, ext * 1.5, 400)
    prof = conc(model, xs, 0.0, z, t, P) * 1e3
    f1 = go.Figure(go.Scatter(x=xs, y=prof, mode="lines"))
    f1.add_hline(y=thr, line_dash="dash", line_color="red", annotation_text="Threshold")
    f1.update_layout(title="Centreline concentration vs downwind distance", xaxis_title="Distance (m)",
                     yaxis_title="mg/m³", yaxis_type="log", height=420)
    c1.plotly_chart(f1, width="stretch")
    if "Plume" in model:
        sy, sz = sigmas(xs, cls, ter)
        f2 = go.Figure([go.Scatter(x=xs, y=sy, name="σy"), go.Scatter(x=xs, y=sz, name="σz")])
        f2.update_layout(title="Dispersion coefficients", xaxis_title="Distance (m)", yaxis_title="m", height=420)
    else:
        ts = np.linspace(10, 7200, 150)
        ct = [float(conc(model, rx, ry, z, tt, P)) * 1e3 for tt in ts]
        f2 = go.Figure(go.Scatter(x=ts, y=ct, mode="lines"))
        f2.add_hline(y=thr, line_dash="dash", line_color="red")
        f2.add_vline(x=t, line_dash="dot")
        f2.update_layout(title="Concentration at receptor vs time", xaxis_title="Time (s)",
                         yaxis_title="mg/m³", height=420)
    c2.plotly_chart(f2, width="stretch")

with tab3:
    st.markdown(r"""
**Gaussian plume (continuous, steady):**
$C=\dfrac{Q}{2\pi u\sigma_y\sigma_z}\exp\!\left(-\dfrac{y^2}{2\sigma_y^2}\right)\left[e^{-\frac{(z-H)^2}{2\sigma_z^2}}+e^{-\frac{(z+H)^2}{2\sigma_z^2}}\right]$

**Gaussian puff (instantaneous):**
$C=\dfrac{m}{(2\pi)^{3/2}\sigma_x\sigma_y\sigma_z}\exp\!\left(-\dfrac{(x-ut)^2}{2\sigma_x^2}-\dfrac{y^2}{2\sigma_y^2}\right)\left[\dots\right]$, with $\sigma_x=\sigma_y$.

**Finite-duration release:** a continuous release for *T* seconds is treated as many small puffs (superposition).

σ values: Briggs correlations (rural/urban) for Pasquill classes A–F.
**Assumptions:** flat terrain, constant wind, no chemical reaction or deposition, passive (neutrally buoyant) gas, ground reflection included. Not valid for dense-gas releases or wind speed < ~1 m/s.
""")