import streamlit as st
st.set_page_config(page_title="6‑Vertex + XFOIL Demo", layout="wide")

import os, subprocess, shutil, tempfile, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from shapely.geometry import Polygon, LineString
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (needed by mpl)
from paper_plane_sim import AeroParams, PaperPlaneSim

# ─────────────────────────────
# 0)  Windows 용 XFOIL 경로
# ─────────────────────────────
XFOIL_EXE = r"C:\Users\Neo\Desktop\XFOIL6.99\xfoil.exe"
xfoil_path = r"C:\Users\Neo\Desktop\XFOIL6.99\xfoil.exe"
xfoil_dir = os.path.dirname(xfoil_path)
# ─────────────────────────────
# 1)  유틸리티
# ─────────────────────────────
def save_foil(fname, xy):
    """xy : (N,2)  ->  .dat"""
    with open(fname, "w") as f:
        f.write("AutoFoil\n")
        for x, y in xy:
            f.write(f"{x: .6f} {y: .6f}\n")

def call_xfoil(foil_path, alpha0=-5, alpha1=10, Re=5e4):
    """XFOIL batch 실행 → (CLα, CD0)  또는 RuntimeError"""
    if not shutil.which(XFOIL_EXE):
        raise RuntimeError("XFOIL 경로가 잘못되었습니다")

    script = "\n".join([
        f"LOAD {foil_path}",
        "PANE",           # panel‑refinement
        "OPER",
        f"VISC {int(Re)}",
        f"ASEQ {alpha0} {alpha1} 1",
        "PACC",
        "polar.dat",
        "",
        "QUIT", ""
    ])

    with tempfile.TemporaryDirectory() as td:
        # foil.dat 과 batch 스크립트 임시폴더에 복사
        tmp_foil = os.path.join(td, "foil.dat")
        shutil.copy(foil_path, tmp_foil)
        # run
        proc = subprocess.run([XFOIL_EXE, "-b"],
                              input=script,
                              text=True,
                              capture_output=True,
                              cwd=td)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "XFOIL fail")

        pol = os.path.join(td, "polar.dat")
        if not os.path.exists(pol):
            raise RuntimeError("polar.dat 생성 실패")

        a, cl, cd = [], [], []
        for ln in open(pol):
            sp = ln.split()
            if len(sp) == 9 and sp[0].replace("-","").replace(".","").isdigit():
                aa, cc, dd = map(float, sp[:3])
                a.append(aa); cl.append(cc); cd.append(dd)
        if len(a) < 4:
            raise RuntimeError("polar 데이터가 충분하지 않음")

        slope, *_ = linregress(a, cl)
        return slope, min(cd)

# ─── 본문 ────────────────────────────────────
st.title("종이비행기 비행 시뮬레이터")
st.markdown("만든이: 김성래 OwO")
st.markdown("브라운이는 최고다! 브라운 브라운~~ ")
st.markdown("사이드바에서 파라미터를 조정한 뒤 ▶ 시뮬레이션 시작을 눌러주세요.")



# ───────────── 사이드바 입력 ─────────────────────


with st.sidebar:
    st.header("Input Parameters")
    # 기본 shape 계수
    shape_dict = {
        "A": {"AR":1.28, "CLα":0.10,  "CD0":0.030},
        "B": {"AR":1.50, "CLα":0.11,  "CD0":0.032},
        "C": {"AR":1.20, "CLα":0.095, "CD0":0.028},
        "D": {"AR":1.00, "CLα":0.085, "CD0":0.035},
        "E": {"AR":1.40, "CLα":0.120, "CD0":0.030},
        "F": {"AR":1.60, "CLα":0.115, "CD0":0.030},
        "G": {"AR":0.90, "CLα":0.090, "CD0":0.040},
    }
    wing_shape = st.selectbox("형상 A~G 선택", list(shape_dict), key="shape")
    params     = shape_dict[wing_shape]

    # Custom XFOIL 모드
    custom = st.checkbox(
        "▶ Custom XFOIL 분석",
        value=False,
        key="custom_xfoil_checkbox"
    )
    st.session_state['custom'] = custom

    run_xfoil_flag = st.sidebar.checkbox("XFOIL 계수 추출", value=False, key="xfoil_flag")



    # 3) 형상/초기조건 (앞각 φ, 뒷각 ψ 는 front_angle/back_angle 으로 통일)
    thk = st.sidebar.slider("두께 t (m)", 0.002, 0.050, 0.02, step=0.002, key="thk")
    W     = st.slider("날개폭 W (m)",    0.10, 0.40, 0.16, 0.01, key="W")
    L     = st.slider("비행길이 L (m)", 0.10, 0.60, 0.25, 0.01, key="L")
    front_angle = st.slider(
        "앞각 φ (°)",
        min_value=0, max_value=90,
        value=st.session_state.get("front_angle", 45),
        key="front_angle",
    )
    back_angle  = st.slider(
        "뒷각 ψ (°)",
        min_value=-90, max_value=90,
        value=st.session_state.get("back_angle", 0),
        key="back_angle",
    )
    mass  = st.number_input("질량 m (kg)", 0.005, 0.050, 0.010, 0.001, key="mass")
    aoa   = st.slider("받음각 AOA (°)",  0.0, 15.0, 5.0, step=0.1, key="aoa")
    init_h= st.slider("초기고도 y₀ (m)", 0.0,  5.0, 0.0, step=0.1, key="init_h")
    v0    = st.slider("발사속도 V₀ (m/s)",2.0, 15.0,10.0, step=0.1, key="v0")
    theta0= st.slider("발사각도 θ₀ (°)",   0,   45,   15,           key="theta0")

    run_btn  = st.button("▶ 시뮬레이션 시작", key="run")
    show_3d  = st.button("▶ 3D View",         key="3d")


    # 특정 시간 힘 분석용 슬라이더
    st.markdown("---")
    if 'traj' in st.session_state and 'sim' in st.session_state:
        times = np.linspace(
            0,
            len(st.session_state['traj']) * st.session_state['sim'].dt,
            len(st.session_state['traj'])
        )
        t_query = st.slider(
            "시간 선택 (초) for Force Analysis",
            0.0, float(times[-1]), float(times[-1]/2),
            step=0.1,
            key="t_query"
        )

    # session_state 한 번에 저장 
    st.session_state.update({
        'wing_shape':  wing_shape,
        'params':      params,
        'custom':      custom,
    })



# ─── 6-Vertex 실루엣 그리기 ──────────────────────


# 슬라이더가 아직 한 번도 호출되지 않았다면 여기서 기본값을 꺼내 쓰도록!
# ---------------------------------------------------
# 6‑Vertex 실루엣 계산  (P1~P6 → poly, poly2d, up/low)
# ---------------------------------------------------
import numpy as np
from shapely.geometry import Polygon

# ① Slider 값(또는 session_state) → W,L,φ,ψ
W = st.session_state.get("W", 0.16)
L = st.session_state.get("L", 0.25)
φ = st.session_state.get("front_angle", 45)
ψ = st.session_state.get("back_angle",  0)

# ② 앞·뒷 여유 거리
d_front = (W/2) / np.tan(np.radians(φ/2)) if abs(φ) > 1e-3 else 0.0
d_back  = (W/2) / np.tan(np.radians(ψ/2)) if abs(ψ) > 1e-3 else 0.0

# ③ 6‑Vertex 좌표
P1 = (-W/2, +L/2 - d_front)
P2 = ( 0.0, +L/2)
P3 = (+W/2, +L/2 - d_front)
P4 = (+W/2, -L/2 + d_back)
P5 = ( 0.0, -L/2)
P6 = (-W/2, -L/2 + d_back)

# 시각화용: 마지막 점(P1) 다시 넣어서 닫힌 루프
poly = np.vstack([P1, P2, P3, P4, P5, P6, P1])    # 시각화용 (닫힌 루프)
pts  = np.vstack([P1, P2, P3, P4, P5, P6])        # 계산용 (중복점 제외)

# shapely Polygon 생성 (중복점 제외)
from shapely.geometry import Polygon
poly2d = Polygon(pts)   # 또는 Polygon(poly[:-1])로 해도 동일

# 시각화용 축 범위 계산
x_min, x_max = pts[:,0].min(), pts[:,0].max()
y_min, y_max = pts[:,1].min(), pts[:,1].max()
x_pad = (x_max - x_min) * 0.1
y_pad = (y_max - y_min) * 0.1

coords_3d = np.array(poly2d.exterior.coords)

thk = 0.02
coords = np.array(poly2d.exterior.coords)  # 이미 닫혀 있음
upper = np.c_[coords, np.full(len(coords), +thk/2)]
lower = np.c_[coords, np.full(len(coords), -thk/2)]

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(projection='3d')

# 위/아래 윤곽선
ax.plot(upper[:,0], upper[:,1], upper[:,2], 'b-', linewidth=2)
ax.plot(lower[:,0], lower[:,1], lower[:,2], 'b-', linewidth=2)

# 측면 연결선
for (xu,yu,zu),(xl,yl,zl) in zip(upper, lower):
    ax.plot([xu,xl], [yu,yl], [zu,zl], color='gray', alpha=0.5)

# ⑤ Extrude 상·하부 좌표 (up, low)
#     poly2d.exterior.coords → (마지막 점은 시작점 반복이므로 제외)
coords_flat = np.array(poly2d.exterior.coords)[:-1]    # (N,2)

up  = np.c_[coords_flat,  np.full(len(coords_flat),  +thk/2)]
low = np.c_[coords_flat,  np.full(len(coords_flat),  -thk/2)]

col_l, col_r = st.columns([1,1])
with col_l:
    fig_s, ax_s = plt.subplots(figsize=(5,4))
    ax_s.plot(poly[:,0], poly[:,1], '-o', lw=2)
    ax_s.set_aspect('equal','box')
    ax_s.set_xlim(x_min-x_pad, x_max+x_pad)
    ax_s.set_ylim(y_min-y_pad, y_max+y_pad)
    ax_s.set_title(f"6-Vertex Silhouette (W={W:.2f}, L={L:.2f})")
    st.pyplot(fig_s, use_container_width=True)

with col_r:
    st.markdown("#### 설계 참고 형상 (A~G)")
    st.image("20250502_221356.png", use_container_width=True)



# ─────────────────────────────
# 4)  3D Extrude 시각화
# ─────────────────────────────


fig3d = plt.figure(figsize=(5,4))
ax3   = fig3d.add_subplot(111, projection="3d")
for (xu,yu,zu),(xl,yl,zl) in zip(up, low):
    ax3.plot([xu,xl],[yu,yl],[zu,zl], color="gray", alpha=0.6)
ax3.plot(up [:,0], up [:,1], up [:,2], 'b-')
ax3.plot(low[:,0], low[:,1], low[:,2], 'b-')
st.pyplot(fig3d, use_container_width=True)


# ─────────────────────────────
# 5)  단면 (Fuselage Root) → foil.dat
#     => X=0 라인 절단
# ─────────────────────────────
root_line = LineString([(0,-L/2-0.1), (0,L/2+0.1)])
cut = poly2d.intersection(root_line)
if cut.is_empty or not isinstance(cut, LineString):
    st.error("X=0 절단면을 얻지 못했습니다 – φ/ψ 각도를 조정해 보세요.")
    st.stop()

# cut.coords: 두 점 (y_min,y_max)  →  chord 길이
y1,y2 = [pt[1] for pt in cut.coords]
chord = abs(y2 - y1)
# 간단한 박스형 에어포일:  x 축은 1→0,  y 는 ±t/2 의 직선 (실제로는 더 정교한 fit 필요)
xs = np.linspace(1, 0, 60)
t  = 0.04                     # ▲ 4 % 두께
ys = 5 * t * (xs*(1-xs))      # 간단한 포물선 배분
foil_xy = (
    list(zip(xs,  ys)) +      # 상면
    list(zip(xs[::-1], -ys[::-1]))  # 하면
)
save_foil("root.foil", foil_xy)


# ─────────────────────────────
# 6)  XFOIL 실행 & 결과
# ─────────────────────────────
if run_xfoil_flag:
    try:
        cla, cd0 = call_xfoil("root.foil")
        st.success(f"XFOIL OK  →  CLα={cla:.3f}, CD0={cd0:.4f}")
    except Exception as e:
        st.error(f"XFOIL 실패:\n{e}")
else:
    st.info("‘XFOIL 계수 추출’ 체크박스를 켜면 CLα, CD0 계산을 시도합니다.")






#==================== 시뮬레이션의 영역 =========================== 