import numpy as np

class AeroParams:
    def __init__(self, S=0.04, AR=1.28, CL_alpha=0.1, CD0=0.03, e=0.8):
        self.S = S
        self.AR = AR
        self.CL_alpha = CL_alpha
        self.CD0 = CD0
        self.e = e

    def coeffs(self, aoa_deg):
        #Lift coefficient, Drag coefficient
        CL = self.CL_alpha * aoa_deg
        CD = self.CD0 + CL**2 / (np.pi * self.AR * self.e)
        return CL, CD

class PaperPlaneSim:
    def __init__(self,
                 mass: float,
                 aero: AeroParams,
                 v0: float,
                 theta0_deg: float,
                 aoa_deg: float,
                 init_h: float = 0.0,
                 dt: float = 0.005,
                 max_steps: int = 5000):
        #인자 그대로 속성에 저장
        self.m         = mass
        self.aero      = aero
        self.v0        = v0
        self.theta0    = np.radians(theta0_deg)
        self.aoa_deg   = aoa_deg
        self.init_h    = init_h
        self.dt        = dt
        self.max_steps = max_steps
        self.rho       = 1.225
        self.g         = 9.81

    def _deriv(self, state):
        x, y, vx, vy = state
        v = np.hypot(vx, vy)
        if v < 1e-6:
            #속도가 거의 0이면 중력만 작용
            return np.array([0.0, 0.0, 0.0, -self.g])

        #단위 속도 벡터
        v_unit   = np.array([vx, vy]) / v
        #리프트 방향: 속도에 수직 (왼쪽으로 90° 회전)
        lift_dir = np.array([-v_unit[1], v_unit[0]])

        #공력 계수
        CL, CD  = self.aero.coeffs(self.aoa_deg)
        q       = 0.5 * self.rho * v**2
        L_force = q * self.aero.S * CL
        D_force = q * self.aero.S * CD

        #힘 분해 (x, y)
        fx = -D_force * v_unit[0] + L_force * lift_dir[0]
        fy = -self.m * self.g    - D_force * v_unit[1] + L_force * lift_dir[1]

        return np.array([vx, vy, fx / self.m, fy / self.m])

    def run(self):
        #초기 속도 벡터
        vx0   = self.v0 * np.cos(self.theta0)
        vy0   = self.v0 * np.sin(self.theta0)
        state = np.array([0.0, self.init_h, vx0, vy0])
        traj  = [state.copy()]

        for _ in range(self.max_steps):
            deriv = self._deriv(state)
            state = state + self.dt * deriv
            #땅 아래로 내려가면 멈춤
            if state[1] < 0:
                break
            traj.append(state.copy())

        return np.array(traj)
