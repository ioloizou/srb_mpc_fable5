"""
Single Rigid Body (SRB) convex MPC for a humanoid (Unitree G1 class).

Model
-----
The robot is approximated as a single rigid body (trunk + lumped limbs) driven by
ground reaction forces (GRFs) applied at discrete contact points. Each foot is
modeled with TWO contact points (heel + toe). This captures the line-foot of a
humanoid: the heel/toe force pair produces the sagittal ankle torque that a point
foot model cannot, which is what lets a biped balance in single support.

State (13):   x = [roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz, g]
Inputs (12):  u = [f_LH, f_LT, f_RH, f_RT]  (3D force per contact point, world frame)

Linearized (about yaw, small roll/pitch, ignoring w x Iw) continuous dynamics:

    theta_dot = Rz(psi)^T * w
    p_dot     = v
    w_dot     = I_w^{-1} * sum_i [r_i]x * f_i
    v_dot     = sum_i f_i / m + [0,0,g]
    g_dot     = 0

with I_w = Rz * I_body * Rz^T and r_i = p_contact_i - p_com. Discretized with a
matrix exponential per horizon step, condensed into a dense QP over the GRFs:

    min_U  (X - Xref)^T Qbar (X - Xref) + U^T Rbar U
    s.t.   friction pyramid  |fx| <= mu*fz, |fy| <= mu*fz
           fz_min <= fz <= fz_max   (fz forced to 0 for swing feet)

This is the MIT convex MPC formulation (Di Carlo et al., 2018) adapted to a
biped with heel/toe contacts.

The demo at the bottom walks the SRB model in a nonlinear rigid body simulation
with a phase-based gait scheduler and Raibert-style foot placement.

Only numpy/scipy needed. Uses OSQP if installed, otherwise a built-in
OSQP-style ADMM solver.

On a real G1 the MPC output maps to stance-leg joint torques via
tau = J_i^T * (-f_i) per contact point (plus swing-leg PD tracking).
"""

import numpy as np
from scipy.linalg import expm, cho_factor, cho_solve

try:
    import osqp
    from scipy import sparse
    HAVE_OSQP = True
except ImportError:
    HAVE_OSQP = False


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rot_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_from_R(R):
    """ZYX Euler angles (roll, pitch, yaw) from rotation matrix."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def solve_qp_admm(P, q, A, l, u, max_iter=1200, rho=0.02, sigma=1e-6,
                  alpha=1.6, warm=None, tol=1e-5):
    """
    Minimal OSQP-style ADMM: min 0.5 x'Px + q'x  s.t. l <= Ax <= u.
    Over-relaxation (alpha) is applied on the Ax term inside the z/y updates
    only. Returns (x, warm_state) for warm-starting the next solve.
    """
    n, m = P.shape[0], A.shape[0]
    Kf = cho_factor(P + sigma * np.eye(n) + rho * (A.T @ A))
    if warm is None:
        x, z, y = np.zeros(n), np.zeros(m), np.zeros(m)
    else:
        x, z, y = (w.copy() for w in warm)
    for it in range(max_iter):
        x = cho_solve(Kf, sigma * x - q + A.T @ (rho * z - y))
        Ax = A @ x
        Ah = alpha * Ax + (1 - alpha) * z
        z_prev = z
        z = np.clip(Ah + y / rho, l, u)
        y = y + rho * (Ah - z)
        if it % 25 == 0:
            prim = np.abs(Ax - np.clip(Ax, l, u)).max()
            dual = rho * np.abs(A.T @ (z - z_prev)).max()
            if prim < tol and dual < tol:
                break
    return x, (x, z, y)


# ----------------------------------------------------------------------------
# Robot / MPC parameters (Unitree G1-ish)
# ----------------------------------------------------------------------------

class G1Params:
    mass = 35.0                                   # kg (G1 ~35 kg)
    inertia = np.diag([1.70, 1.60, 0.60])         # body-frame lumped inertia, kg m^2
    com_height = 0.69                             # nominal CoM height, m
    hip_offset_y = 0.12                           # lateral hip offset, m
    foot_half_len = 0.09                          # heel/toe offset from foot center, m
    mu = 0.5                                      # friction coefficient
    fz_min = 5.0                                  # N per contact point when in stance
    fz_max = 350.0                                # N per contact point
    g = 9.81


class MPCParams:
    dt = 0.04                # MPC discretization (s)
    horizon = 10             # 0.4 s lookahead
    # State weights: [rpy(3), pos(3), omega(3), vel(3), g]
    Q = np.array([500, 500, 250,   300, 300, 1200,   8, 8, 15,   30, 30, 40,   0.0])
    R = 5e-6                 # force regularization


# ----------------------------------------------------------------------------
# The MPC
# ----------------------------------------------------------------------------

NUM_CP = 4        # contact points: L-heel, L-toe, R-heel, R-toe
NX, NU = 13, 3 * NUM_CP


class SRBMPC:
    def __init__(self, robot: G1Params, params: MPCParams):
        self.rb = robot
        self.pp = params
        self._last_U = None

    # -- dynamics ------------------------------------------------------------

    def _continuous(self, yaw, r_cp):
        """A_c, B_c for given yaw and contact-point positions rel. to CoM."""
        rb = self.rb
        Rz = rot_z(yaw)
        Iw = Rz @ rb.inertia @ Rz.T
        Iw_inv = np.linalg.inv(Iw)

        A = np.zeros((NX, NX))
        A[0:3, 6:9] = Rz.T          # euler rates ~ Rz(psi)^T * omega
        A[3:6, 9:12] = np.eye(3)    # p_dot = v
        A[11, 12] = 1.0             # vz_dot += g   (state g = -9.81)

        B = np.zeros((NX, NU))
        for i in range(NUM_CP):
            B[6:9, 3*i:3*i+3] = Iw_inv @ skew(r_cp[i])
            B[9:12, 3*i:3*i+3] = np.eye(3) / rb.mass
        return A, B

    def _discretize(self, A, B, dt):
        M = np.zeros((NX + NU, NX + NU))
        M[:NX, :NX] = A
        M[:NX, NX:] = B
        Md = expm(M * dt)
        return Md[:NX, :NX], Md[:NX, NX:]

    # -- solve ---------------------------------------------------------------

    def solve(self, x0, contact_table, cp_pos_table, x_ref_table):
        """
        x0             : (13,) current state
        contact_table  : (N, 4) bool, contact point in stance at step k
        cp_pos_table   : (N, 4, 3) world positions of contact points at step k
        x_ref_table    : (N, 13) reference states
        returns        : (4, 3) forces for the first step, per contact point
        """
        rb, pp = self.rb, self.pp
        N = pp.horizon

        # Per-step discrete dynamics (B depends on foot positions vs. ref CoM).
        Ad, Bd = [], []
        for k in range(N):
            r_cp = cp_pos_table[k] - x_ref_table[k, 3:6]
            A_c, B_c = self._continuous(x0[2], r_cp)
            Adk, Bdk = self._discretize(A_c, B_c, pp.dt)
            Ad.append(Adk)
            Bd.append(Bdk)

        # Condense: X = A_qp x0 + B_qp U
        A_qp = np.zeros((N * NX, NX))
        B_qp = np.zeros((N * NX, N * NU))
        A_pow = np.eye(NX)
        for k in range(N):
            A_pow = Ad[k] @ A_pow
            A_qp[k*NX:(k+1)*NX] = A_pow
            blk = Bd[k]
            for j in range(k, -1, -1):
                B_qp[k*NX:(k+1)*NX, j*NU:(j+1)*NU] = blk if j == k else \
                    Ad[k] @ B_qp[(k-1)*NX:k*NX, j*NU:(j+1)*NU]

        # Keep only the force variables of contact points that are actually in
        # stance somewhere in the horizon; swing forces are identically zero,
        # so we delete their columns instead of adding equality constraints.
        active = []          # flat variable indices into U
        for k in range(N):
            for i in range(NUM_CP):
                if contact_table[k, i]:
                    active.extend(range(k * NU + 3*i, k * NU + 3*i + 3))
        active = np.array(active, dtype=int)
        n_act = len(active)
        if n_act == 0:
            return np.zeros((NUM_CP, 3))
        B_act = B_qp[:, active]

        Qbar = np.kron(np.eye(N), np.diag(pp.Q))
        x_ref = x_ref_table.reshape(-1)
        P = B_act.T @ Qbar @ B_act + pp.R * np.eye(n_act)
        P = 0.5 * (P + P.T)
        qv = B_act.T @ Qbar @ (A_qp @ x0 - x_ref)

        # Constraints per active contact point:
        # |fx| <= mu fz, |fy| <= mu fz, fz_min <= fz <= fz_max
        mu = rb.mu
        C1 = np.array([[1, 0, -mu],
                       [-1, 0, -mu],
                       [0, 1, -mu],
                       [0, -1, -mu],
                       [0, 0, 1.0]])
        n_forces = n_act // 3
        A_con = np.kron(np.eye(n_forces), C1)
        l_con = np.full(A_con.shape[0], -np.inf)
        u_con = np.zeros(A_con.shape[0])
        l_con[4::5] = rb.fz_min
        u_con[4::5] = rb.fz_max

        if HAVE_OSQP:
            prob = osqp.OSQP()
            prob.setup(sparse.csc_matrix(P), qv, sparse.csc_matrix(A_con),
                       l_con, u_con, verbose=False, warm_start=True)
            U_act = prob.solve().x
        else:
            warm = getattr(self, "_warm", None)
            if warm is not None and warm[0].shape[0] != n_act:
                warm = None
            U_act, self._warm = solve_qp_admm(P, qv, A_con, l_con, u_con,
                                             warm=warm)

        U = np.zeros(N * NU)
        U[active] = U_act
        return U[:NU].reshape(NUM_CP, 3)


# ----------------------------------------------------------------------------
# Gait scheduler + swing legs (for the demo)
# ----------------------------------------------------------------------------

class GaitScheduler:
    """
    Walking gait scheduled in integer MPC ticks so that contact transitions
    land exactly on MPC solve boundaries (no stale-force windows).
    Default: period 20 ticks (0.8 s), duty 12 (0.48 s stance),
    half-period offset -> 2 ticks (0.08 s) of double support per step.
    """

    def __init__(self, dt_tick, period_ticks=20, duty_ticks=12):
        self.dt = dt_tick
        self.period = period_ticks
        self.duty = duty_ticks
        self.offsets = [0, period_ticks // 2]   # [left, right]

    def in_stance(self, tick, foot):
        return (tick + self.offsets[foot]) % self.period < self.duty

    def contact_table(self, tick, N):
        """(N, 4) contact point schedule over the horizon."""
        table = np.zeros((N, NUM_CP), dtype=bool)
        for k in range(N):
            for foot in range(2):
                s = self.in_stance(tick + k, foot)
                table[k, 2*foot] = s
                table[k, 2*foot+1] = s
        return table

    def swing_progress(self, tick_f, foot):
        """None if in stance, else swing progress s in [0,1). tick_f: float."""
        ph = (tick_f + self.offsets[foot]) % self.period
        if ph < self.duty:
            return None
        return (ph - self.duty) / (self.period - self.duty)

    def stance_time(self):
        return self.duty * self.dt


def raibert_target(p_com, v_com, v_cmd, yaw, foot, rb: G1Params, gait):
    """Raibert heuristic + capture-point feedback footstep target."""
    sign = 1.0 if foot == 0 else -1.0
    hip = p_com + rot_z(yaw) @ np.array([0.0, sign * rb.hip_offset_y, 0.0])
    k_cap = np.sqrt(rb.com_height / rb.g)
    p = hip[:2] + v_com[:2] * gait.stance_time() / 2.0 \
        + k_cap * (v_com[:2] - v_cmd[:2])
    # Clamp step reach
    d = p - hip[:2]
    r = np.linalg.norm(d)
    if r > 0.35:
        p = hip[:2] + d / r * 0.35
    return np.array([p[0], p[1], 0.0])


def swing_pos(p_liftoff, p_target, s, height=0.08):
    """Cycloid-ish swing interpolation, s in [0,1]."""
    s = np.clip(s, 0.0, 1.0)
    xy = p_liftoff[:2] + (p_target[:2] - p_liftoff[:2]) * \
        (s - np.sin(2*np.pi*s) / (2*np.pi))
    z = height * 0.5 * (1 - np.cos(2*np.pi*s)) if s < 1.0 else 0.0
    return np.array([xy[0], xy[1], z])


def foot_contact_points(p_foot, yaw, rb: G1Params):
    """Heel/toe world positions for a foot at p_foot with body yaw."""
    d = rot_z(yaw) @ np.array([rb.foot_half_len, 0.0, 0.0])
    return np.array([p_foot - d, p_foot + d])


# ----------------------------------------------------------------------------
# Demo: nonlinear SRB simulation walking under the MPC
# ----------------------------------------------------------------------------

def run_demo(t_end=8.0, v_cmd_xy=(0.4, 0.0), yaw_rate_cmd=0.0, plot=True):
    rb, pp = G1Params(), MPCParams()
    mpc = SRBMPC(rb, pp)

    dt_sim = 0.002
    mpc_decimation = int(round(pp.dt / dt_sim))
    gait = GaitScheduler(pp.dt)

    # Sim state (nonlinear rigid body)
    p = np.array([0.0, 0.0, rb.com_height])
    R = np.eye(3)
    v = np.zeros(3)
    w = np.zeros(3)

    # Feet
    foot_pos = np.array([[0.0,  rb.hip_offset_y, 0.0],
                         [0.0, -rb.hip_offset_y, 0.0]])
    liftoff = foot_pos.copy()
    target = foot_pos.copy()
    was_stance = [True, True]

    v_cmd = np.array([v_cmd_xy[0], v_cmd_xy[1], 0.0])
    yaw_ref = 0.0
    yaw_unwrapped, prev_yaw = 0.0, 0.0
    forces = np.zeros((NUM_CP, 3))

    log = {k: [] for k in ["t", "p", "v", "rpy", "fz", "contact"]}

    n_steps = int(t_end / dt_sim)
    for step in range(n_steps):
        t = step * dt_sim
        tick = step // mpc_decimation          # integer MPC tick
        tick_f = step / mpc_decimation         # fractional tick for swing
        rpy = rpy_from_R(R)
        dyaw = (rpy[2] - prev_yaw + np.pi) % (2*np.pi) - np.pi
        yaw_unwrapped += dyaw
        prev_yaw = rpy[2]
        yaw = yaw_unwrapped

        # ---- gait bookkeeping / swing legs
        for foot in range(2):
            sp = gait.swing_progress(tick_f, foot)
            if sp is None:                              # stance
                if not was_stance[foot]:                # touchdown
                    foot_pos[foot] = np.array([target[foot][0],
                                               target[foot][1], 0.0])
                was_stance[foot] = True
            else:                                       # swing
                if was_stance[foot]:                    # liftoff
                    liftoff[foot] = foot_pos[foot].copy()
                    was_stance[foot] = False
                v_cmd_w = rot_z(yaw) @ v_cmd
                target[foot] = raibert_target(p, v, v_cmd_w, yaw,
                                              foot, rb, gait)
                foot_pos[foot] = swing_pos(liftoff[foot], target[foot], sp)

        # ---- MPC at 25 Hz (every tick)
        if step % mpc_decimation == 0:
            N = pp.horizon
            contact_table = gait.contact_table(tick, N)

            # Reference: integrate the (body-frame) command from the current
            # position along the commanded heading.
            x_ref_table = np.zeros((N, NX))
            p_xy = np.array([p[0], p[1]])
            for k in range(N):
                yaw_k = yaw_ref + yaw_rate_cmd * (k + 1) * pp.dt
                v_k = rot_z(yaw_k) @ v_cmd
                p_xy = p_xy + v_k[:2] * pp.dt
                x_ref_table[k, 2] = yaw_k
                x_ref_table[k, 3:5] = p_xy
                x_ref_table[k, 5] = rb.com_height
                x_ref_table[k, 8] = yaw_rate_cmd
                x_ref_table[k, 9:12] = v_k
                x_ref_table[k, 12] = -rb.g

            # Predicted contact point positions over the horizon:
            # currently-stancing feet stay put; feet that (re)land during the
            # horizon appear at their Raibert target.
            cp_pos_table = np.zeros((N, NUM_CP, 3))
            for k in range(N):
                for foot in range(2):
                    if gait.in_stance(tick + k, foot) and \
                       gait.in_stance(tick, foot):
                        pf = foot_pos[foot]
                    else:
                        pf = target[foot]
                    cps = foot_contact_points(
                        np.array([pf[0], pf[1], 0.0]), yaw, rb)
                    cp_pos_table[k, 2*foot] = cps[0]
                    cp_pos_table[k, 2*foot+1] = cps[1]

            x0 = np.concatenate([[rpy[0], rpy[1], yaw], p, w, v, [-rb.g]])
            forces = mpc.solve(x0, contact_table, cp_pos_table, x_ref_table)
            yaw_ref += yaw_rate_cmd * pp.dt

        # ---- apply forces to nonlinear SRB dynamics
        in_stance = np.array([gait.in_stance(tick, i // 2)
                              for i in range(NUM_CP)])
        f_total = np.zeros(3)
        tau_total = np.zeros(3)
        for i in range(NUM_CP):
            if not in_stance[i]:
                continue
            foot = i // 2
            cps = foot_contact_points(foot_pos[foot], rpy_from_R(R)[2], rb)
            r = cps[i % 2] - p
            f_total += forces[i]
            tau_total += np.cross(r, forces[i])

        Iw = R @ rb.inertia @ R.T
        a = f_total / rb.mass + np.array([0, 0, -rb.g])
        w_dot = np.linalg.solve(Iw, tau_total - np.cross(w, Iw @ w))

        p = p + v * dt_sim
        v = v + a * dt_sim
        R = expm(skew(w * dt_sim)) @ R
        w = w + w_dot * dt_sim

        log["t"].append(t)
        log["p"].append(p.copy())
        log["v"].append(v.copy())
        log["rpy"].append(rpy_from_R(R))
        log["fz"].append(forces[:, 2].copy())
        log["contact"].append(in_stance.copy())

        if p[2] < 0.3 or abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
            print(f"FELL at t={t:.2f}s"); break

    for k in log:
        log[k] = np.array(log[k])

    v_avg = log["v"][len(log["v"])//2:, 0].mean()
    print(f"Done. avg vx over 2nd half: {v_avg:.3f} m/s (cmd {v_cmd[0]:.2f}), "
          f"final height {log['p'][-1, 2]:.3f} m, "
          f"|roll|max {np.abs(log['rpy'][:,0]).max():.3f} rad")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        ax[0].plot(log["t"], log["v"][:, 0], label="vx")
        ax[0].plot(log["t"], log["v"][:, 1], label="vy")
        ax[0].axhline(v_cmd[0], ls="--", c="gray", label="vx cmd")
        ax[0].set_ylabel("CoM vel [m/s]"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(log["t"], log["p"][:, 2], label="z")
        ax[1].plot(log["t"], log["rpy"][:, 0], label="roll")
        ax[1].plot(log["t"], log["rpy"][:, 1], label="pitch")
        ax[1].axhline(rb.com_height, ls="--", c="gray")
        ax[1].set_ylabel("height / rpy"); ax[1].legend(); ax[1].grid(alpha=.3)
        labels = ["L heel", "L toe", "R heel", "R toe"]
        for i in range(NUM_CP):
            ax[2].plot(log["t"], log["fz"][:, i], label=labels[i], lw=0.8)
        ax[2].set_ylabel("fz [N]"); ax[2].set_xlabel("t [s]")
        ax[2].legend(ncol=4); ax[2].grid(alpha=.3)
        fig.suptitle("SRB-MPC humanoid walking (G1-class)")
        fig.tight_layout()
        fig.savefig("srb_mpc_walk.png", dpi=130)
        print("Saved plot to srb_mpc_walk.png")

    return log


if __name__ == "__main__":
    run_demo(t_end=8.0, v_cmd_xy=(0.4, 0.0))
