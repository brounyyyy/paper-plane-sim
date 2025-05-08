import streamlit as st
st.set_page_config(page_title="6-Vertex + XFOIL Demo", layout="wide")

# ---------- 일반 모듈 ----------
import os, math, shutil, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")        # Streamlit에만 그림 표시
import matplotlib.pyplot as plt
plt.ioff()          # GUI 창 끄기
plt.close('all')    # 혹시 떠 있던 창 제거
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from mpl_toolkits.mplot3d import Axes3D             # noqa: F401
from scipy.stats import linregress
import os, subprocess, shutil, tempfile, math
import matplotlib.pyplot as plt
from paper_plane_sim import AeroParams, PaperPlaneSim

def save_foil(path: str, xy, name: str = "AutoFoil"):
    """
    • path : 저장할 파일 경로(예: 'root.foil' 혹은 'foil.dat')
    • xy   : (N,2) 형식의 numpy 배열 또는 리스트  [[x0,y0], [x1,y1], ...]
    • name : XFOIL 첫 줄에 들어갈 에어포일 이름
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy 는 (N,2) 배열이어야 합니다.")

    with open(path, "w", encoding="utf-8") as fp:
        fp.write(f"{name}\n")          # 1줄 : 이름
        for x, y in xy:                # 좌표(마지막 점 반복 X)
            fp.write(f"{x:.6f} {y:.6f}\n")

# ---------- 외부 시뮬레이터 ----------
# 없어도 앱은 켜집니다.  (모듈이 있다면 자동 사용)
try:
    from paper_plane_sim import AeroParams, PaperPlaneSim
    SIM_AVAILABLE = True
except ImportError:
    SIM_AVAILABLE = False

import os, shutil, subprocess, tempfile
from   scipy.stats import linregress

# 반드시 한곳에서만 정의!
XFOIL_EXE = r"C:\Users\Neo\Desktop\XFOIL6.99\xfoil.exe"

def run_xfoil(foil_path, alpha_range=(-10,15), Re=100000):
    alpha0, alpha1 = alpha_range
    with tempfile.TemporaryDirectory() as td:
        foil_dat = os.path.join(td, "foil.dat")
        polar_file = os.path.join(td, "polar.dat")
        shutil.copy(foil_path, foil_dat)

        script = "\n".join([
            f"LOAD {os.path.basename(foil_dat)}",
            "PANE",
            "OPER",
            f"VISC {int(Re)}",
            "PACC",
            "polar.dat",     # ← 파일 열기
            "",              # ← 빈 줄로 PACC 헤더 종료
            f"ASEQ {alpha_range[0]} {alpha_range[1]} 1",
            "",              # ← 빈 줄 : PACC 닫기
            "QUIT",
            ""
        ])


        proc = subprocess.run([XFOIL_EXE],
                              input=script, text=True,
                              capture_output=True, cwd=td)

        if proc.returncode != 0:
            raise RuntimeError(f"[XFOIL 실패] 코드 {proc.returncode}\n\n"
                               f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

        if not os.path.exists(polar_file):
            raise RuntimeError("polar.dat 가 생성되지 않았습니다.\n"
                               f"STDOUT:\n{proc.stdout}")

        # --- polar.dat 파싱 ---
        alphas, cls, cds = [], [], []
        with open(polar_file) as f:
            for line in f:
                # 헤더/공백/주석(#) 줄은 skip
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                try:
                    a, cl, cd = map(float, parts[:3])
                    alphas.append(a)
                    cls.append(cl)
                    cds.append(cd)
                except ValueError:
                    # 숫자로 변환되지 않는 헤더 줄은 무시
                    continue


        if len(alphas) < 4:
            raise RuntimeError(f"polar.dat 데이터가 부족합니다 (읽은 각도 {len(alphas)})\n"
                               f"STDOUT:\n{proc.stdout}")

        slope, *_ = linregress(alphas, cls)
        return slope, min(cds)


        return dcl_dalpha, cd_min


# ─── 본문 ────────────────────────────────────
st.title("종이비행기 비행 시뮬레이터")
st.markdown("만든이: 김성래 OwO")
st.markdown("브라운이는 최고다! 브라운 브라운~~ ")
st.markdown("사이드바에서 파라미터를 조정한 뒤 ▶ 시뮬레이션 시작을 눌러주세요.")

# ===============================================================
# 2)  사이드바 ‑ 입력값
# ===============================================================
with st.sidebar:
    st.header("종이 비행기 디자인")
    if st.button("Matplotlib 창/메모리 리셋"):
        plt.close('all')
        st.success("모든 Matplotlib 핸들을 닫았습니다.")

    # (1) 기본 형상별 초기 계수
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

    # (2) 6‑Vertex 파라미터
    W   = st.slider("날개폭  W (m)", 0.10, 0.40, 0.16, 0.01, key="W")
    L   = st.slider("비행길이 L (m)",0.10, 0.60, 0.25, 0.01, key="L")
    φ   = st.slider("앞각 φ", 0,  90, 45, key="front_angle")
    ψ   = st.slider("뒷각 ψ", -90, 90, 0, key="back_angle")

    thk = st.slider("종이 두께 t (m)", 0.002, 0.050, 0.020, 0.002, key="thk")

    # (3) XFOIL 실행 플래그
    run_xfoil_flag = st.checkbox("XFOIL 계수 추출", value=False)

    # (4) 시뮬레이션용 입력 (옵션)
    if SIM_AVAILABLE:
        st.header("비행 초기조건 설정")
        mass  = st.number_input("질량 m (kg)", 0.005, 0.050, 0.010, 0.001, key="mass")
        aoa   = st.slider("받음각 AOA (°)",  0.0, 15.0, 5.0, step=0.1, key="aoa")
        init_h = st.slider("초기고도 y₀ (m)",     0.0, 5.0, 0.0, 0.1)
        v0     = st.slider("발사속도 V₀ (m/s)",   2.0,15.0,10.0,0.1)
        theta0 = st.slider("발사각도 θ₀ (°)",      0,  45, 15)

        run_btn = st.button("▶ 시뮬레이션 시작")
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
    st.write("DEBUG φ, ψ  :", st.session_state.get("front_angle"),
                               st.session_state.get("back_angle"))

# ===============================================================
# 3)  6‑Vertex 실루엣 생성 & 2D/3D 시각화
# ===============================================================
# (3‑1) 꼭짓점 계산
def six_vertices(W,L,φ,ψ):
    d_f = (W/2)/math.tan(math.radians(φ/2)) if φ else 0
    d_b = (W/2)/math.tan(math.radians(ψ/2)) if ψ else 0
    return np.array([
        [-W/2, +L/2 - d_f],
        [  0.0, +L/2   ],
        [+W/2, +L/2-d_f],
        [+W/2, -L/2+d_b],
        [  0.0, -L/2   ],
        [-W/2, -L/2+d_b]
    ])

verts2d = six_vertices(W,L,φ,ψ)
poly2d  = Polygon(verts2d)
closed  = np.vstack([verts2d, verts2d[0]])   # 시각화용 루프

# ─────────────────────────────
# 6-Vertex 실루엣 시각화 + 이미지 표시
# ─────────────────────────────
import numpy as np
from shapely.geometry import Polygon

col_l, col_r = st.columns([1, 1])  # 1:1 비율 컬럼

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

coords_flat = np.array(poly2d.exterior.coords)[:-1]   # (N,2) ← 첫 줄
save_foil("root.foil", coords_flat)                   # ← 둘째 줄

# -------------- XFOIL 해석 --------------------
if run_xfoil_flag:                               # ← 체크박스가 True 일 때만
    # 1) root.foil 저장
    coords_flat = np.array(poly2d.exterior.coords)[:-1]
    save_foil("root.foil", coords_flat)

    # 2) 실행 (캐시를 쓰면 한번만 돈다)
    with st.spinner("Running XFOIL…"):
        try:
            cl_alpha, cd0 = run_xfoil("root.foil",
                                      alpha_range=(-10, 15),
                                      Re=100_000)
        except Exception as e:
            st.error(f"XFOIL 실패:\n{e}")
            cl_alpha = cd0 = None
        else:
            st.success("XFOIL 완료")
            st.write(f"**CLα = {cl_alpha:.4f} [1/rad]**")
            st.write(f"**CD₀ = {cd0:.4f}**")
else:
    st.info("‘XFOIL 계수 추출’ 체크박스를 켜면 CLα, CD0를 계산합니다.")


# 시각화용 축 범위 계산
x_min, x_max = pts[:,0].min(), pts[:,0].max()
y_min, y_max = pts[:,1].min(), pts[:,1].max()
x_pad = (x_max - x_min) * 0.1
y_pad = (y_max - y_min) * 0.1

# 왼쪽: 6-Vertex 실루엣 플롯
with col_l:
    fig_s, ax_s = plt.subplots(figsize=(5, 4))
    ax_s.plot(poly[:, 0], poly[:, 1], '-o', lw=2)
    ax_s.set_aspect('equal', 'box')
    ax_s.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_s.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_s.set_title(f"6-Vertex Silhouette (W={W:.2f}, L={L:.2f})")
    ax_s.set_xlabel("X (m)")
    ax_s.set_ylabel("Y (m)")
    st.pyplot(fig_s, use_container_width=True)

# 오른쪽: 형상 참고 이미지
with col_r:
    st.markdown("#### 설계 참고 형상 (A~G)")
    st.image("20250502_221356.png", use_container_width=True)

# (3‑3) 3D Extrude
coords = np.array(poly2d.exterior.coords)[:-1]   # 중복점 제거
up  = np.c_[coords, np.full(len(coords), +thk/2)]
low = np.c_[coords, np.full(len(coords), -thk/2)]

fig3d = plt.figure(figsize=(4, 4), dpi=100)
fig3d = plt.figure(figsize=(4,4)); ax3d = fig3d.add_subplot(111,projection='3d')
ax3d.plot(up[:,0], up[:,1], up[:,2],'b-')
ax3d.plot(low[:,0],low[:,1],low[:,2],'b-')
for (u,l) in zip(up,low):
    ax3d.plot([u[0],l[0]],[u[1],l[1]],[u[2],l[2]],color='gray',alpha=0.5)
ax3d.set_title("Extruded Plate"); ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
st.pyplot(fig3d, use_container_width=False)


# ===============================================================
# 4)  root 단면( X=0 절단 ) → foil.dat  → XFOIL
# ===============================================================
root_line = LineString([(0, -L), (0, L)])
cut = poly2d.intersection(root_line)
if cut.is_empty or not isinstance(cut, LineString):
    st.warning("X=0 절단이 없습니다(φ, ψ 각이 특이).")
else:
    y1,y2 = [pt[1] for pt in cut.coords]
    chord = abs(y2-y1)
    # ‑‑ 간단 평판 에어포일 (xs=1→0→1 / ys=0)
    xs = np.linspace(1,0,40)
    ys = np.zeros_like(xs)

    # 6) XFOIL 실행 & 결과
    if run_xfoil_flag:
        try:
            # 기존 call_xfoil → run_xfoil 로 변경
            cla, cd0 = run_xfoil("root.foil")      # ← 함수 이름만 교체
            st.success(f"XFOIL OK  →  CLα={cla:.3f}, CD0={cd0:.4f}")
        except Exception as e:
            st.error(f"XFOIL 실패:\n{e}")
    else:
        st.info("‘XFOIL 계수 추출’ 체크박스를 켜면 CLα, CD0 계산을 시도합니다.")






#==================== 시뮬레이션의 영역 ===========================
if run_btn:
    S = W * L
    AR = W**2 / S
    aero = AeroParams(S=S, AR=params["AR"],
                      CL_alpha=params["CLα"],
                      CD0=params["CD0"], e=0.8)

    sim = PaperPlaneSim(mass, aero,
                        v0, theta0, aoa,
                        init_h=init_h, dt=0.005, max_steps=5000)
    traj = sim.run()
    st.session_state['traj'] = traj
    st.session_state['sim'] = sim  # 추가 저장

    # 결과
    rng = traj[-1, 0]
    tof = (len(traj) - 1) * sim.dt
    st.subheader("시뮬레이션 결과")
    st.write(f"- Flight Range: **{rng:.2f} m**")
    st.write(f"- Time Aloft: **{tof:.2f} s**")
    st.write(f"- Shape Code: **{wing_shape}**")

if 'traj' in st.session_state:
    traj = st.session_state['traj']
    sim  = st.session_state['sim']
    dt   = sim.dt
    times = np.linspace(0, len(traj)*dt, len(traj))
    vx, vy = traj[:, 2], traj[:, 3]
    speed = np.sqrt(vx**2 + vy**2)

    # 상단 2D 궤적
    fig2d, ax2d = plt.subplots(figsize=(6, 4))
    ax2d.plot(traj[:, 0], traj[:, 1], color="tab:blue", label="Trajectory")

    step_interval = int(0.1 / dt)
    dot_indices = np.arange(0, len(traj), step_interval)
    ax2d.scatter(traj[dot_indices, 0], traj[dot_indices, 1],
                 color="red", s=20, label="Every 0.1s")
    ax2d.set_xlabel("Distance (m)")
    ax2d.set_ylabel("Altitude (m)")
    ax2d.set_title("2D Flight Trajectory")
    ax2d.grid(True)
    ax2d.legend()
    st.pyplot(fig2d, use_container_width=True)

    # 상단 Plotly 3D 애니메이션
    import plotly.graph_objects as go
    if st.button("Show 3D Flight Animation"):
        fig_anim = go.Figure(frames=[
            go.Frame(data=[go.Scatter3d(
                x=traj[:k+1, 0],
                y=np.zeros_like(traj[:k+1, 0]),
                z=traj[:k+1, 1],
                mode='lines+markers',
                line=dict(width=5, color='orange'),
                marker=dict(size=3, color='orange')
            )], name=str(k))
            for k in range(5, len(traj), 5)
        ])
        fig_anim.add_trace(go.Scatter3d(
            x=[traj[0, 0]],
            y=[0],
            z=[traj[0, 1]],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(width=5, color='orange')
        ))
        fig_anim.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Z (m)',
                zaxis_title='Y (Altitude)'
            ),
            title="3D Flight Animation",
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play", method="animate",
                              args=[None, dict(frame=dict(duration=30, redraw=True),
                                               fromcurrent=True)])]
            )],
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_anim, use_container_width=True)

    # 하단 3열 시각화
    col1, col2, col3 = st.columns(3)

    # 가속도
    with col1:
        acc = np.array([sim._deriv(state)[2:] for state in traj])
        fig_acc, ax_acc = plt.subplots(figsize=(5, 4))
        ax_acc.plot(times, acc[:, 0], label="Ax (m/s²)", color="green")
        ax_acc.plot(times, acc[:, 1], label="Ay (m/s²)", color="purple")
        ax_acc.set_xlabel("Time (s)")
        ax_acc.set_ylabel("Acceleration (m/s²)")
        ax_acc.set_title("Acceleration vs Time")
        ax_acc.grid(True)
        ax_acc.legend()
        st.pyplot(fig_acc, use_container_width=True)

    # 속도
    with col2:
        fig_vel, ax_vel = plt.subplots(figsize=(5, 4))
        ax_vel.plot(times, vx, label="Vx (m/s)", color="orange")
        ax_vel.plot(times, vy, label="Vy (m/s)", color="blue")
        ax_vel.plot(times, speed, label="Speed", color="black", linestyle="--")
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_vel.set_title("Velocity vs Time")
        ax_vel.grid(True)
        ax_vel.legend()
        st.pyplot(fig_vel, use_container_width=True)

    # 에너지
    with col3:
        g = 9.81
        KE = 0.5 * mass * speed**2
        PE = mass * g * traj[:, 1]
        E_total = KE + PE
        fig_E, ax_E = plt.subplots(figsize=(5, 4))
        ax_E.plot(times, KE, label="Kinetic", color="cyan")
        ax_E.plot(times, PE, label="Potential", color="magenta")
        ax_E.plot(times, E_total, label="Total", color="black", linestyle="--")
        ax_E.set_xlabel("Time (s)")
        ax_E.set_ylabel("Energy (J)")
        ax_E.set_title("Mechanical Energy vs Time")
        ax_E.grid(True)
        ax_E.legend()
        st.pyplot(fig_E, use_container_width=True)

        # 시간축, 속도, 위치 등 계산
    times = np.linspace(0, len(traj) * dt, len(traj))
    vx, vy = traj[:, 2], traj[:, 3]
    speed  = np.hypot(vx, vy)
    g      = 9.81
    mass   = sim.m
    S      = sim.aero.S

    # 에너지 계산 및 손실률 정의
    KE       = 0.5 * mass * speed**2
    PE       = mass * g * traj[:, 1]
    E_total  = KE + PE
    loss_ratio = (E_total[0] - E_total) / E_total[0] * 100

    # ───────── 1행: 에너지 손실률 / 고도 변화 ─────────
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Energy Loss Ratio Over Time")
        fig_loss, ax_loss = plt.subplots(figsize=(5, 4))
        ax_loss.plot(times, loss_ratio, color="darkred")
        ax_loss.set_xlabel("Time (s)")
        ax_loss.set_ylabel("Energy Loss (%)")
        ax_loss.set_title("Energy Loss Ratio")
        ax_loss.grid(True)
        st.pyplot(fig_loss, use_container_width=True)

    with row1_col2:
        st.subheader("Altitude Over Time")
        fig_alt, ax_alt = plt.subplots(figsize=(5, 4))
        ax_alt.plot(times, traj[:, 1], color="royalblue")
        ax_alt.set_xlabel("Time (s)")
        ax_alt.set_ylabel("Altitude (m)")
        ax_alt.set_title("Altitude vs Time")
        ax_alt.grid(True)
        st.pyplot(fig_alt, use_container_width=True)




    # 3D 궤적
    fig3d = plt.figure(figsize=(6,4))
    ax3d = fig3d.add_subplot(111, projection='3d')
    zs = np.zeros_like(traj[:,0])
    ax3d.plot(traj[:,0], zs, traj[:,1], color="tab:orange")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Z (m)")
    ax3d.set_zlabel("Y (m)")
    ax3d.set_title("3D Flight Trajectory (Z=0)")
    ax3d.text2D(
        0.02, 0.90,
        f"Mass: {mass:.3f} kg | AOA: {aoa:.1f}° | Init Alt: {init_h:.2f} m | "
        f"Speed: {v0:.2f} m/s | Angle: {theta0}°",
        transform=ax3d.transAxes, va="top", ha="left", fontsize=10
    )
    st.pyplot(fig3d, use_container_width=False)



# ───── Force 분석 시각화 (슬라이더는 이미 sidebar에 있음) ─────
if 'traj' in st.session_state and 'sim' in st.session_state and 't_query' in st.session_state:
    traj = st.session_state['traj']
    sim  = st.session_state['sim']
    dt   = sim.dt
    t_query = st.session_state['t_query']

    # session_state에서 변수 로드
    W = st.session_state['W']
    L = st.session_state['L']
    aoa = st.session_state['aoa']
    mass = st.session_state['mass']
    params = st.session_state['params']

    # 시간에 해당하는 인덱스 및 상태
    idx = min(int(t_query / dt), len(traj)-1)
    x, y, vx, vy = traj[idx]
    v = np.hypot(vx, vy)

    # 공력계수 및 힘 계산
    rho = 1.225
    q = 0.5 * rho * v**2
    S_local = W * L
    AR_local = W**2 / S_local
    CL = params["CLα"] * aoa
    CD = params["CD0"] + CL**2 / (np.pi * AR_local * 0.8)
    Lift = q * S_local * CL
    Drag = q * S_local * CD
    Gravity = mass * 9.81
    Thrust = 0.0

    forces = {
        "Thrust": Thrust,
        "Drag": Drag,
        "Lift": Lift,
        "Gravity": Gravity
    }
    dirs = {
        "Thrust": np.array([0, +1]),
        "Drag": -np.array([vx, vy]) / v if v > 0 else np.array([1, 0]),
        "Lift": np.array([-vy, vx]) / v if v > 0 else np.array([0, 1]),
        "Gravity": np.array([0, -1])
    }
    colmap = {
        "Thrust": "gray", "Drag": "red", "Lift": "blue", "Gravity": "black"
    }

    col1, col2 = st.columns(2)

    # ─── 좌측: Force 막대그래프
    with col1:
        st.subheader(f"Force Magnitudes at t = {t_query:.2f} s")
        fig_fm, ax_fm = plt.subplots(figsize=(5, 4))
        ax_fm.bar(forces.keys(), forces.values(), color=colmap.values())
        ax_fm.set_ylabel("Force (N)")
        ax_fm.set_title("Forces at That Moment")
        ax_fm.grid(True, axis='y')
        st.pyplot(fig_fm, use_container_width=True)

    # ─── 우측: Force 벡터 시각화
    with col2:
        st.subheader(f"Force Vector Directions at t = {t_query:.2f} s")
        fig_vd, ax_vd = plt.subplots(figsize=(5, 4))
        offset = 0.3
        max_force = max(abs(f) for f in forces.values())
        scale = 0.8 / max_force if max_force > 0 else 0.0

        ax_vd.plot(traj[:, 0], traj[:, 1], '--', color='lightgray')
        ax_vd.scatter(x, y, color='black', s=30, label='Plane')

        for name, force in forces.items():
            origin = np.array([x, y]) + offset * dirs[name]
            vec = dirs[name] * force * scale
            ax_vd.arrow(origin[0], origin[1], vec[0], vec[1],
                        head_width=0.05, color=colmap[name],
                        length_includes_head=True)
            ax_vd.text(origin[0] + vec[0], origin[1] + vec[1],
                       f"{name}\n{force:.1f} N", color=colmap[name],
                       ha='center', va='bottom')

        ax_vd.set_xlim(x - 1, x + 1)
        ax_vd.set_ylim(y - 1, y + 1)
        ax_vd.set_aspect('equal')
        ax_vd.set_xlabel("X (m)")
        ax_vd.set_ylabel("Y (m)")
        ax_vd.set_title("2D Force Vectors")
        ax_vd.grid(True)
        st.pyplot(fig_vd, use_container_width=True)



    #  Plotly 3D (인터랙티브) #아니 진짜왜 이게 작동해야 슬라읻가 작동하는거야 으아아아
    if show_3d:
        fig = go.Figure(
            go.Scatter3d(
                x=traj[:,0],
                y=np.zeros_like(traj[:,0]),
                z=traj[:,1],
                mode='lines',
                line=dict(color='orange', width=4),
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis_title='Distance (m)',
                yaxis_title='Z (m)',
                zaxis_title='Altitude (m)',
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            title="Interactive 3D Flight Trajectory",
        )
        st.plotly_chart(fig, use_container_width=True)


