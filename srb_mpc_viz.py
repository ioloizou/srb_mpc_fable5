"""
Live 3D animation for the SRB-MPC humanoid walker (srb_mpc_g1.py).

Visual style follows the classic SRBD visualizer (cyan torso box, red edges,
GRF arrows, foot patches on the ground plane), with a few upgrades:

  * heel AND toe GRF arrows per foot (that's where the MPC forces live)
  * swing-foot trajectory + ghost outline of the upcoming footstep target
  * CoM trace, support-consistent foot coloring (left red / right blue)
  * camera that follows the robot
  * HUD with time / commanded vs actual velocity
  * LIVE TELEOP: arrow keys change the walking command while it runs
        up/down    : +/- 0.1 m/s forward
        left/right : +/- 0.2 rad/s turn
        w/s        : +/- 0.05 m/s lateral
        space      : stop (zero all commands)

Run live:            python3 srb_mpc_viz.py
Save a clip instead: python3 srb_mpc_viz.py --save out.gif --t-end 4

Requires srb_mpc_g1.py in the same directory (only numpy/scipy/matplotlib).
"""

import argparse
import numpy as np
from scipy.linalg import expm

from srb_mpc_g1 import (G1Params, MPCParams, SRBMPC, GaitScheduler, NUM_CP,
                        NX, raibert_target, swing_pos, foot_contact_points,
                        rot_z, rpy_from_R, skew)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D
from matplotlib.animation import FuncAnimation, PillowWriter


# ----------------------------------------------------------------------------
# Simulation stepper (same logic as srb_mpc_g1.run_demo, exposed per-tick)
# ----------------------------------------------------------------------------

class WalkingSim:
    def __init__(self, v_cmd_xy=(0.4, 0.0), yaw_rate_cmd=0.0):
        self.rb, self.pp = G1Params(), MPCParams()
        self.mpc = SRBMPC(self.rb, self.pp)
        self.dt_sim = 0.002
        self.deci = int(round(self.pp.dt / self.dt_sim))
        self.gait = GaitScheduler(self.pp.dt)

        rb = self.rb
        self.p = np.array([0.0, 0.0, rb.com_height])
        self.R = np.eye(3)
        self.v = np.zeros(3)
        self.w = np.zeros(3)

        self.foot_pos = np.array([[0.0,  rb.hip_offset_y, 0.0],
                                  [0.0, -rb.hip_offset_y, 0.0]])
        self.liftoff = self.foot_pos.copy()
        self.target = self.foot_pos.copy()
        self.was_stance = [True, True]

        self.v_cmd = np.array([v_cmd_xy[0], v_cmd_xy[1], 0.0])
        self.yaw_rate_cmd = yaw_rate_cmd
        self.yaw_ref = 0.0
        self.yaw_unwrapped, self.prev_yaw = 0.0, 0.0
        self.forces = np.zeros((NUM_CP, 3))
        self.step_count = 0
        self.fallen = False

    @property
    def t(self):
        return self.step_count * self.dt_sim

    def _substep(self):
        rb, pp, gait = self.rb, self.pp, self.gait
        step = self.step_count
        tick = step // self.deci
        tick_f = step / self.deci

        rpy = rpy_from_R(self.R)
        dyaw = (rpy[2] - self.prev_yaw + np.pi) % (2*np.pi) - np.pi
        self.yaw_unwrapped += dyaw
        self.prev_yaw = rpy[2]
        yaw = self.yaw_unwrapped

        # gait bookkeeping / swing legs
        for foot in range(2):
            sp = gait.swing_progress(tick_f, foot)
            if sp is None:
                if not self.was_stance[foot]:
                    self.foot_pos[foot] = np.array([self.target[foot][0],
                                                    self.target[foot][1], 0.0])
                self.was_stance[foot] = True
            else:
                if self.was_stance[foot]:
                    self.liftoff[foot] = self.foot_pos[foot].copy()
                    self.was_stance[foot] = False
                v_cmd_w = rot_z(yaw) @ self.v_cmd
                self.target[foot] = raibert_target(self.p, self.v, v_cmd_w,
                                                   yaw, foot, rb, gait)
                self.foot_pos[foot] = swing_pos(self.liftoff[foot],
                                                self.target[foot], sp)

        # MPC solve on tick boundaries
        if step % self.deci == 0:
            N = pp.horizon
            contact_table = gait.contact_table(tick, N)

            x_ref_table = np.zeros((N, NX))
            p_xy = self.p[:2].copy()
            for k in range(N):
                yaw_k = self.yaw_ref + self.yaw_rate_cmd * (k + 1) * pp.dt
                v_k = rot_z(yaw_k) @ self.v_cmd
                p_xy = p_xy + v_k[:2] * pp.dt
                x_ref_table[k, 2] = yaw_k
                x_ref_table[k, 3:5] = p_xy
                x_ref_table[k, 5] = rb.com_height
                x_ref_table[k, 8] = self.yaw_rate_cmd
                x_ref_table[k, 9:12] = v_k
                x_ref_table[k, 12] = -rb.g

            cp_pos_table = np.zeros((N, NUM_CP, 3))
            for k in range(N):
                for foot in range(2):
                    if gait.in_stance(tick + k, foot) and \
                       gait.in_stance(tick, foot):
                        pf = self.foot_pos[foot]
                    else:
                        pf = self.target[foot]
                    cps = foot_contact_points(
                        np.array([pf[0], pf[1], 0.0]), yaw, rb)
                    cp_pos_table[k, 2*foot] = cps[0]
                    cp_pos_table[k, 2*foot+1] = cps[1]

            x0 = np.concatenate([[rpy[0], rpy[1], yaw],
                                 self.p, self.w, self.v, [-rb.g]])
            self.forces = self.mpc.solve(x0, contact_table,
                                         cp_pos_table, x_ref_table)
            self.yaw_ref += self.yaw_rate_cmd * pp.dt
            # keep the yaw reference from drifting far from reality
            self.yaw_ref = yaw + ((self.yaw_ref - yaw + np.pi)
                                  % (2*np.pi) - np.pi)

        # apply forces to the nonlinear SRB dynamics
        in_stance = np.array([gait.in_stance(tick, i // 2)
                              for i in range(NUM_CP)])
        f_total, tau_total = np.zeros(3), np.zeros(3)
        for i in range(NUM_CP):
            if not in_stance[i]:
                continue
            foot = i // 2
            cps = foot_contact_points(self.foot_pos[foot],
                                      rpy_from_R(self.R)[2], rb)
            r = cps[i % 2] - self.p
            f_total += self.forces[i]
            tau_total += np.cross(r, self.forces[i])

        Iw = self.R @ rb.inertia @ self.R.T
        a = f_total / rb.mass + np.array([0, 0, -rb.g])
        w_dot = np.linalg.solve(Iw, tau_total - np.cross(self.w, Iw @ self.w))

        self.p = self.p + self.v * self.dt_sim
        self.v = self.v + a * self.dt_sim
        self.R = expm(skew(self.w * self.dt_sim)) @ self.R
        self.w = self.w + w_dot * self.dt_sim
        self.step_count += 1

        if self.p[2] < 0.3 or abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
            self.fallen = True

    def step_tick(self):
        """Advance one MPC tick (0.04 s)."""
        tick = self.step_count // self.deci
        # contacts are constant within a tick; record them for this tick so
        # the drawn stance flags match the forces solved for it
        self.in_stance = [self.gait.in_stance(tick, f) for f in range(2)]
        for _ in range(self.deci):
            if not self.fallen:
                self._substep()


# ----------------------------------------------------------------------------
# Graphics helpers
# ----------------------------------------------------------------------------

TORSO_HALF = np.array([0.13, 0.11, 0.26])    # half extents of the torso box
FOOT_HALF = np.array([0.11, 0.035])          # half extents of a foot patch
FORCE_SCALE = 1.0 / 400.0                    # m of arrow per N

_BOX_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1)
                         for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
_BOX_FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
              [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]


def torso_faces(p, R):
    corners = (_BOX_CORNERS * TORSO_HALF) @ R.T + p
    return [corners[f] for f in _BOX_FACES]


def foot_polygon(center, yaw, z=None):
    c = np.array([[sx, sy] for sx, sy in
                  [(-1, -1), (1, -1), (1, 1), (-1, 1)]]) * FOOT_HALF
    Rz2 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = c @ Rz2.T + center[:2]
    zz = center[2] if z is None else z
    return [np.column_stack([xy, np.full(4, zz + 0.002)])]


# ----------------------------------------------------------------------------
# Animator
# ----------------------------------------------------------------------------

class SRBDAnimator:
    FOOT_COLORS = ["#c23b3b", "#2b4fc2"]         # left red, right blue

    def __init__(self, sim: WalkingSim):
        self.sim = sim
        self.fig = plt.figure(figsize=(11, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        ax = self.ax
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.set_box_aspect((2.5, 2.4, 1.6))

        # torso
        self.torso = Poly3DCollection(
            torso_faces(sim.p, sim.R), facecolors="cyan", alpha=0.55,
            edgecolors="salmon", linewidths=1.2)
        ax.add_collection3d(self.torso)

        # feet + footstep-target ghosts
        self.feet, self.ghosts = [], []
        for f in range(2):
            patch = Poly3DCollection(
                foot_polygon(sim.foot_pos[f], 0.0),
                facecolors=self.FOOT_COLORS[f], alpha=0.9, edgecolors="k")
            ax.add_collection3d(patch)
            self.feet.append(patch)
            ghost, = ax.plot([], [], [], ls="--", lw=1.0,
                             color=self.FOOT_COLORS[f], alpha=0.7)
            self.ghosts.append(ghost)

        # CoM trace + vertical drop line
        self.trace, = ax.plot([], [], [], color="teal", lw=1.2, alpha=0.8)
        self.drop, = ax.plot([], [], [], color="gray", lw=0.8, ls=":")
        self.trace_pts = []

        self.quivers = []
        self.hud = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                             family="monospace", fontsize=10)
        ax.text2D(0.02, 0.02,
                  "teleop: arrows = fwd/turn, w/s = lateral, space = stop",
                  transform=ax.transAxes, fontsize=8, color="gray")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event):
        s = self.sim
        if event.key == "up":
            s.v_cmd[0] = min(s.v_cmd[0] + 0.1, 1.0)
        elif event.key == "down":
            s.v_cmd[0] = max(s.v_cmd[0] - 0.1, -0.4)
        elif event.key == "left":
            s.yaw_rate_cmd = min(s.yaw_rate_cmd + 0.2, 1.0)
        elif event.key == "right":
            s.yaw_rate_cmd = max(s.yaw_rate_cmd - 0.2, -1.0)
        elif event.key == "w":
            s.v_cmd[1] = min(s.v_cmd[1] + 0.05, 0.3)
        elif event.key == "s":
            s.v_cmd[1] = max(s.v_cmd[1] - 0.05, -0.3)
        elif event.key == " ":
            s.v_cmd[:] = 0.0
            s.yaw_rate_cmd = 0.0

    def update(self, _frame):
        s = self.sim
        s.step_tick()
        yaw = rpy_from_R(s.R)[2]

        self.torso.set_verts(torso_faces(s.p, s.R))

        for f in range(2):
            self.feet[f].set_verts(foot_polygon(s.foot_pos[f], yaw))
            self.feet[f].set_alpha(0.9 if s.in_stance[f] else 0.45)
            if not s.in_stance[f]:      # ghost of the planned touchdown
                g = foot_polygon(s.target[f], yaw, z=0.0)[0]
                g = np.vstack([g, g[0]])
                self.ghosts[f].set_data_3d(g[:, 0], g[:, 1], g[:, 2])
            else:
                self.ghosts[f].set_data_3d([], [], [])

        # GRF arrows at heel/toe of stance feet
        for q in self.quivers:
            q.remove()
        self.quivers.clear()
        for i in range(NUM_CP):
            foot = i // 2
            if not s.in_stance[foot]:
                continue
            cp = foot_contact_points(s.foot_pos[foot], yaw, s.rb)[i % 2]
            fvec = s.forces[i] * FORCE_SCALE
            if np.linalg.norm(fvec) < 0.02:
                continue
            self.quivers.append(self.ax.quiver(
                cp[0], cp[1], cp[2], fvec[0], fvec[1], fvec[2],
                color=self.FOOT_COLORS[foot], lw=1.8,
                arrow_length_ratio=0.12))

        self.trace_pts.append(s.p.copy())
        if len(self.trace_pts) > 400:
            self.trace_pts.pop(0)
        tr = np.array(self.trace_pts)
        self.trace.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])
        self.drop.set_data_3d([s.p[0]] * 2, [s.p[1]] * 2, [0, s.p[2]])

        # camera follows the robot
        self.ax.set_xlim(s.p[0] - 1.2, s.p[0] + 1.3)
        self.ax.set_ylim(s.p[1] - 1.2, s.p[1] + 1.2)
        self.ax.set_zlim(0, 1.6)

        v_body = rot_z(yaw).T @ s.v
        status = "FALLEN" if s.fallen else "walking"
        self.hud.set_text(
            f"t = {s.t:5.2f} s   [{status}]\n"
            f"cmd: fwd {s.v_cmd[0]:+.2f}  lat {s.v_cmd[1]:+.2f} m/s"
            f"  yaw {s.yaw_rate_cmd:+.1f} rad/s\n"
            f"act: fwd {v_body[0]:+.2f}  lat {v_body[1]:+.2f} m/s"
            f"   z {s.p[2]:.2f} m")
        return []

    def run_live(self):
        self.anim = FuncAnimation(self.fig, self.update,
                                  interval=int(self.sim.pp.dt * 1000),
                                  blit=False, cache_frame_data=False)
        plt.show()

    def save(self, path, t_end=4.0, fps=25):
        n = int(t_end / self.sim.pp.dt)
        anim = FuncAnimation(self.fig, self.update, frames=n, blit=False)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
        print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--yaw-rate", type=float, default=0.0)
    ap.add_argument("--save", type=str, default=None,
                    help="save a GIF instead of showing live")
    ap.add_argument("--t-end", type=float, default=4.0)
    args = ap.parse_args()

    sim = WalkingSim((args.vx, args.vy), args.yaw_rate)
    viz = SRBDAnimator(sim)
    if args.save:
        viz.save(args.save, t_end=args.t_end)
    else:
        viz.run_live()
