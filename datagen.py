"""Scripted expert demonstrations for LIBERO-Spatial bowl-to-plate tasks, in Drake.

Each episode:
  1. Scene: load a recorded LIBERO scene (fixture placement, settled object
     heights), then move every free object to a uniformly random x-y position in
     the region LIBERO samples it from. Objects stacked on another object move
     with it. Scenes with new overlaps, or that shift while settling, are rejected.
  2. Grasp: a deterministic top-down grasp of the bowl wall, chosen from the
     bowl pose (the side facing the table's centerline, as in the human demos).
  3. Key frames: pre-grasp, grasp, lift, pre-place, place, retreat.
  4. Inverse kinematics turns each key frame into a Panda configuration.
  5. Motion planning: straight Cartesian approach/lift/descent/retreat, and
     collision-free joint-space paths between them (direct, else RRT-Connect
     with shortcutting), with the grasped bowl attached while carrying.
  6. Execution: the planned gripper poses are tracked with the same 7-D OSC
     delta actions as the LIBERO human data, stepping LiberoEnv.

Only successful episodes are kept. Files use the LIBERO HDF5 layout and names,
so LiberoEnv, `libero.py replay --data <output>`, and HDF5 loaders read them.
"""

import argparse
import json
import math
import re
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from pydrake.all import (
    InverseKinematics,
    IpoptSolver,
    QuaternionFloatingJoint,
    RigidTransform,
    Role,
    RotationMatrix,
)
from scipy.spatial.transform import Rotation

from libero import (
    ASSETS,
    DATA,
    NAMES,
    LiberoEnv,
    load_initial_states,
    load_recording_model,
)

BOWL = "akita_black_bowl_1_main"
PLATE = "plate_1_main"
FREE_OBJECTS = (
    "akita_black_bowl_1",
    "akita_black_bowl_2",
    "cookies_1",
    "glazed_rim_porcelain_ramekin_1",
    "plate_1",
)
DEFAULT_TASKS = (2, 5, 7)  # table center, on the ramekin, on the stove
CONTROL_DT = 0.05  # LiberoEnv.step: 20 Hz actions
POS_SCALE, ROT_SCALE = 0.05, 0.5  # LiberoEnv.step: metres and radians per unit action
OPEN, CLOSE = -1.0, 1.0
# Bowls left of this line are grasped on their +y side, others on their -y
# side: the side facing the table's centerline, matching the human demos.
GRASP_SPLIT_Y = 0.1
GENERATOR_TAG = "drake-scripted-v1"


@dataclass
class ExpertConfig:
    grasp_radius: float = 0.045  # grip site to bowl axis; bowl wall (human mean 0.045)
    grasp_height: float = 0.03  # grip site above the bowl's bottom
    approach: float = 0.10  # pre-grasp height above the grasp
    lift: float = 0.15
    place_height: float = 0.015  # bowl bottom above the plate origin at release
    preplace: float = 0.08
    retreat: float = 0.10
    speed: float = 0.25  # peak gripper speed in free space (m/s)
    fine_speed: float = 0.15  # peak gripper speed near objects (m/s)
    joint_speed: float = 1.0  # rad/s
    turn_speed: float = 1.0  # rad/s of gripper rotation
    lookahead: int = 3  # steps; offsets the OSC's ~0.16 s tracking lag
    close_steps: int = 10
    open_steps: int = 6
    settle_tolerance: float = 0.004  # m, before grasping and releasing
    max_settle_steps: int = 15
    clearance: float = 0.01  # free-space clearance to obstacles
    fine_clearance: float = 0.003
    region_margin: float = 0.0  # widen LIBERO's placement regions (m)
    scene_settle_steps: int = 10  # >= 1
    max_steps: int = 500


class PlanningError(RuntimeError):
    pass


def stacking(task_id, libero_root=ASSETS):
    """{object: support object} from the task's BDDL initial state."""
    path = Path(libero_root) / "libero/libero/bddl_files/libero_spatial" / f"{NAMES[task_id]}.bddl"
    init = path.read_text().split("(:init", 1)[1].split("(:goal", 1)[0]
    pairs = re.findall(r"\(On\s+(\S+)\s+(\S+)\)", init)
    return {a: b for a, b in pairs if a in FREE_OBJECTS and b in FREE_OBJECTS}


def placement_boxes(task_id, model, libero_root=ASSETS):
    """Per-object x-y bounds of LIBERO's placement regions for this task.

    Taken from the support of the 50 official initial states, which LIBERO drew
    uniformly from these regions (BDDL ranges plus its sampler's margins).
    """
    starts = load_initial_states(libero_root, task_id)
    boxes = {}
    for name in FREE_OBJECTS:
        a = 1 + model.jnt_qposadr[model.joint(name + "_joint0").id]
        xy = starts[:, a : a + 2]
        boxes[name] = (xy.min(0).tolist(), xy.max(0).tolist())
    return boxes


def mujoco_state(env):
    """Drake state as LIBERO's flattened MuJoCo [time, qpos, qvel]; inverse of set_state."""
    m, plant, ctx = env.model, env.plant, env.plant_context
    q, v = plant.GetPositions(ctx), plant.GetVelocities(ctx)
    qpos, qvel = m.qpos0.copy(), np.zeros(m.nv)
    for j, joint in env.joints.items():
        qpos[m.jnt_qposadr[j]] = q[joint.position_start()]
        qvel[m.jnt_dofadr[j]] = v[joint.velocity_start()]
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body = env.bodies[int(m.jnt_bodyid[j])]
        X = plant.EvalBodyPoseInWorld(ctx, body)
        V = plant.EvalBodySpatialVelocityInWorld(ctx, body)
        a, d = m.jnt_qposadr[j], m.jnt_dofadr[j]
        qpos[a : a + 3] = X.translation()
        qpos[a + 3 : a + 7] = X.rotation().ToQuaternion().wxyz()
        # MuJoCo free joints: world-frame linear, body-frame angular velocity.
        qvel[d : d + 3] = V.translational()
        qvel[d + 3 : d + 6] = X.rotation().matrix().T @ V.rotational()
    return np.r_[env.context.get_time(), qpos, qvel]


def quat2axisangle(quat):
    """robosuite.utils.transform_utils.quat2axisangle (xyzw), as LIBERO stores ee_ori."""
    w = min(max(quat[3], -1.0), 1.0)
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(w) / den


def proprioception(env, state):
    """LIBERO's low-dimensional observations, via MuJoCo kinematics as robosuite does."""
    m, d = env.model, env.mj_data
    d.qpos[:] = state[1 : 1 + m.nq]
    mujoco.mj_kinematics(m, d)
    eef_pos = d.site_xpos[m.site("gripper0_grip_site").id].copy()
    eef_quat = d.xquat[m.body("robot0_right_hand").id][[1, 2, 3, 0]].copy()  # xyzw
    fingers = [m.jnt_qposadr[m.joint(f"gripper0_finger_joint{i}").id] for i in (1, 2)]
    gripper = d.qpos[fingers].copy()
    joints = d.qpos[[m.jnt_qposadr[m.joint(f"robot0_joint{i}").id] for i in range(1, 8)]].copy()
    ee_ori = quat2axisangle(eef_quat)
    return {
        "ee_pos": eef_pos,
        "ee_ori": ee_ori,
        "ee_states": np.r_[eef_pos, ee_ori],
        "gripper_states": gripper,
        "joint_states": joints,
        "robot_states": np.r_[gripper, eef_pos, eef_quat],
    }


def interpolate(X0, X1, s):
    R0 = X0.rotation().matrix()
    delta = Rotation.from_matrix(R0.T @ X1.rotation().matrix()).as_rotvec()
    R = R0 @ Rotation.from_rotvec(s * delta).as_matrix()
    return RigidTransform(RotationMatrix(R), (1 - s) * X0.translation() + s * X1.translation())


def raised(X, dz):
    return RigidTransform(X.rotation(), X.translation() + [0, 0, dz])


def angle_between(R0, R1):
    return float(np.linalg.norm(Rotation.from_matrix(R0.T @ R1).as_rotvec()))


class Planner:
    """IK, collision checks, and motion planning on a scratch copy of the scene."""

    def __init__(self, env, cfg, rng):
        self.env, self.plant, self.cfg, self.rng = env, env.plant, cfg, rng
        self.root = env.diagram.CreateDefaultContext()
        self.ctx = self.plant.GetMyMutableContextFromRoot(self.root)
        self.tool = env.tool
        self.qi = np.array(env.controller.qi)
        self.lo = self.plant.GetPositionLowerLimits()[self.qi]
        self.hi = self.plant.GetPositionUpperLimits()[self.qi]
        self.nominal = env.controller.initial  # the OSC's nullspace posture
        self.solver = IpoptSolver()
        self.bowl = self.plant.GetBodyByName(BOWL)
        self.quaternion_starts = [
            j.position_start()
            for j in map(self.plant.get_joint, self.plant.GetJointIndices())
            if j.type_name() == QuaternionFloatingJoint.kTypeName
        ]
        self.query = self.plant.get_geometry_query_input_port()
        inspector = self.query.Eval(self.ctx).inspector()
        self.geometry_body = {
            g: self.plant.GetBodyFromFrameId(inspector.GetFrameId(g)).name()
            for g in inspector.GetAllGeometryIds(Role.kProximity)
        }
        names = set(self.geometry_body.values())
        self.robot = {
            n
            for n in names
            if (n.startswith("robot0_") and n not in ("robot0_base", "robot0_link0"))
            or n.startswith("gripper0_")
        }
        # The base link and pedestal never move relative to the table.
        self.ignored = {n for n in names if n.startswith("mount0_")} | {"robot0_link0"}
        self.sync()
        fingers = [self.body_in_tool(f"gripper0_finger_joint{i}_tip") for i in (1, 2)]
        approach = (fingers[0] + fingers[1]) / 2
        self.approach_G = approach / np.linalg.norm(approach)
        across = fingers[1] - fingers[0]
        across -= across.dot(self.approach_G) * self.approach_G
        self.fingers_G = across / np.linalg.norm(across)

    def sync(self):
        """Snapshot the simulator's current positions as the planning scene."""
        q = self.plant.GetPositions(self.env.plant_context).copy()
        # Simulated quaternions drift off unit norm; IK pins them, so normalize.
        for i in self.quaternion_starts:
            q[i : i + 4] /= np.linalg.norm(q[i : i + 4])
        self.q_scene = q
        self.plant.SetPositions(self.ctx, self.q_scene)

    @property
    def q_now(self):
        return self.q_scene[self.qi].copy()

    def body_in_tool(self, name):
        X_WG = self.tool.CalcPoseInWorld(self.ctx)
        p_W = self.plant.EvalBodyPoseInWorld(self.ctx, self.plant.GetBodyByName(name)).translation()
        return X_WG.inverse() @ p_W

    def pose(self, body):
        return self.plant.EvalBodyPoseInWorld(self.env.plant_context, self.plant.GetBodyByName(body))

    def set_arm(self, q_arm):
        q = self.q_scene.copy()
        q[self.qi] = q_arm
        self.plant.SetPositions(self.ctx, q)

    def fk(self, q_arm):
        self.set_arm(q_arm)
        return self.tool.CalcPoseInWorld(self.ctx)

    def top_down(self, side, R_current):
        """Gripper pointing down with the fingers closing along `side`."""

        def frame(x, z):
            return np.column_stack([x, np.cross(z, x), z])

        down = np.array([0.0, 0.0, -1.0])
        options = [frame(s * side, down) @ frame(self.fingers_G, self.approach_G).T for s in (1, -1)]
        return RotationMatrix(min(options, key=lambda R: angle_between(R_current, R)))

    def ik(self, X, seed, nominal=None):
        """Arm configuration placing the grip site at X, closest to `nominal`."""
        ik = InverseKinematics(self.plant, self.ctx, with_joint_limits=True)
        q, prog = ik.q(), ik.get_mutable_prog()
        p = X.translation()
        ik.AddPositionConstraint(
            self.tool, np.zeros(3), self.plant.world_frame(), p - 5e-4, p + 5e-4
        )
        ik.AddOrientationConstraint(
            self.plant.world_frame(), X.rotation(), self.tool, RotationMatrix(), 5e-3
        )
        others = np.setdiff1d(np.arange(len(self.q_scene)), self.qi)
        prog.AddBoundingBoxConstraint(self.q_scene[others], self.q_scene[others], q[others])
        target = self.nominal if nominal is None else nominal
        prog.AddQuadraticErrorCost(np.eye(len(self.qi)), target, q[self.qi])
        guess = self.q_scene.copy()
        guess[self.qi] = seed
        prog.SetInitialGuess(q, guess)
        # SNOPT stalls on this pinned 48-variable program; IPOPT solves it reliably.
        result = self.solver.Solve(prog)
        return result.GetSolution(q)[self.qi] if result.is_success() else None

    def clear(self, q_arm, attached=None, ignore=(), margin=0.0):
        """No moving body (robot, and the attached bowl) within `margin` of an obstacle."""
        self.set_arm(q_arm)
        movers = set(self.robot)
        if attached is not None:
            X_WG = self.tool.CalcPoseInWorld(self.ctx)
            self.plant.SetFreeBodyPose(self.ctx, self.bowl, X_WG @ attached)
            movers.add(BOWL)
        skip = self.ignored | set(ignore)
        query = self.query.Eval(self.ctx)
        for pair in query.ComputeSignedDistancePairwiseClosestPoints(margin):
            a, b = self.geometry_body[pair.id_A], self.geometry_body[pair.id_B]
            if (a in movers) == (b in movers):
                continue  # static pairs, robot self-pairs, and gripper-bowl contact
            if (b if a in movers else a) not in skip:
                return False
        return True

    def edge_clear(self, q0, q1, **check):
        n = max(1, math.ceil(np.max(np.abs(q1 - q0)) / 0.02))
        return all(self.clear(q0 + (q1 - q0) * i / n, **check) for i in range(1, n + 1))

    def straight(self, q0, X1, step=0.01, **check):
        """IK along the straight Cartesian line from FK(q0) to X1."""
        X0 = self.fk(q0)
        n = max(1, math.ceil(np.linalg.norm(X1.translation() - X0.translation()) / step))
        path = [q0]
        for i in range(1, n + 1):
            q = self.ik(interpolate(X0, X1, i / n), path[-1], nominal=path[-1])
            if q is None or not self.clear(q, **check):
                return None
            path.append(q)
        return path

    def connect(self, q0, q1, **check):
        """Collision-free joint-space path: direct if possible, else RRT-Connect."""
        if self.edge_clear(q0, q1, **check):
            return [q0, q1]
        if not self.clear(q1, **check):
            raise PlanningError("goal configuration in collision")
        path = self.rrt_connect(q0, q1, **check)
        if path is None:
            raise PlanningError("RRT-Connect found no path")
        return self.shortcut(path, **check)

    def rrt_connect(self, q0, q1, iterations=3000, step=0.15, **check):
        trees = [([q0], [-1]), ([q1], [-1])]

        def extend(tree, target):
            nodes, parents = tree
            i = int(np.argmin([np.linalg.norm(n - target) for n in nodes]))
            delta = target - nodes[i]
            dist = np.linalg.norm(delta)
            new = target if dist <= step else nodes[i] + delta * step / dist
            if not self.edge_clear(nodes[i], new, **check):
                return None
            nodes.append(new)
            parents.append(i)
            return len(nodes) - 1

        def branch(tree, i):
            nodes, parents = tree
            out = []
            while i >= 0:
                out.append(nodes[i])
                i = parents[i]
            return out

        for it in range(iterations):
            grow, other = (trees[0], trees[1]) if it % 2 == 0 else (trees[1], trees[0])
            i = extend(grow, self.rng.uniform(self.lo, self.hi))
            if i is None:
                continue
            target = grow[0][i]
            while True:
                j = extend(other, target)
                if j is None:
                    break
                if np.allclose(other[0][j], target):
                    a, b = branch(grow, i), branch(other, j)
                    path = a[::-1] + b[1:]
                    return path if grow is trees[0] else path[::-1]
        return None

    def shortcut(self, path, iterations=80, **check):
        path = list(path)
        for _ in range(iterations):
            if len(path) < 3:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i > 1 and self.edge_clear(path[i], path[j], **check):
                path = path[: i + 1] + path[j:]
        return path

    def timed(self, path, speed):
        """Gripper poses at 20 Hz along a joint path, with a minimum-jerk profile."""
        dense = [path[0]]
        for q0, q1 in zip(path, path[1:]):
            n = max(1, math.ceil(np.max(np.abs(q1 - q0)) / 0.01))
            dense += [q0 + (q1 - q0) * i / n for i in range(1, n + 1)]
        poses = [self.fk(q) for q in dense]
        cost = [0.0]
        for k in range(1, len(dense)):
            dp = np.linalg.norm(poses[k].translation() - poses[k - 1].translation())
            dr = angle_between(poses[k - 1].rotation().matrix(), poses[k].rotation().matrix())
            dq = np.max(np.abs(dense[k] - dense[k - 1]))
            c = max(dp / speed, dr / self.cfg.turn_speed, dq / self.cfg.joint_speed)
            cost.append(cost[-1] + c)
        total = cost[-1]
        # A minimum-jerk profile peaks at 1.875x its mean speed.
        steps = max(2, math.ceil(1.875 * total / CONTROL_DT))
        refs = []
        for k in range(1, steps + 1):
            tau = k / steps
            s = total * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
            i = min(int(np.searchsorted(cost, s, side="right")), len(dense) - 1)
            span = cost[i] - cost[i - 1] if i > 0 else 0.0
            u = (s - cost[i - 1]) / span if span > 0 else 1.0
            refs.append(self.fk(dense[i - 1] + (dense[i] - dense[i - 1]) * min(max(u, 0), 1)))
        return refs


class Recorder:
    """Steps LiberoEnv with OSC delta actions and records LIBERO-format data."""

    KEYS = ("actions", "states", "robot_states", "agentview_rgb", "eye_in_hand_rgb")
    PROPRIO = ("ee_pos", "ee_ori", "ee_states", "gripper_states", "joint_states")

    def __init__(self, env, obs, cfg, video=False):
        self.env, self.obs, self.cfg, self.video = env, obs, cfg, video
        self.data = {k: [] for k in self.KEYS + self.PROPRIO}
        self.frames = []

    @property
    def steps(self):
        return len(self.data["actions"])

    def action(self, X, grip):
        pose = self.env.tool.CalcPoseInWorld(self.env.plant_context)
        dp = (X.translation() - pose.translation()) / POS_SCALE
        R = X.rotation().matrix() @ pose.rotation().matrix().T
        dr = Rotation.from_matrix(R).as_rotvec() / ROT_SCALE
        return np.r_[np.clip(np.r_[dp, dr], -1, 1), grip]

    def act(self, X, grip):
        if self.steps >= self.cfg.max_steps:
            raise PlanningError("episode exceeded max_steps")
        a = self.action(X, grip)
        state = mujoco_state(self.env)
        proprio = proprioception(self.env, state)
        self.data["actions"].append(a)
        self.data["states"].append(state)
        for key in self.PROPRIO + ("robot_states",):
            self.data[key].append(proprio[key])
        # LIBERO stores OpenGL-convention (vertically flipped) images.
        self.data["agentview_rgb"].append(self.obs["agentview_rgb"][::-1].copy())
        self.data["eye_in_hand_rgb"].append(self.obs["eye_in_hand_rgb"][::-1].copy())
        if self.video:
            self.frames.append(self.obs["video_rgb"])
        self.obs = self.env.step(a)[0]

    def error(self, X):
        pose = self.env.tool.CalcPoseInWorld(self.env.plant_context)
        return np.linalg.norm(X.translation() - pose.translation()), angle_between(
            pose.rotation().matrix(), X.rotation().matrix()
        )

    def follow(self, refs, grip, settle=True):
        for k in range(len(refs)):
            self.act(refs[min(k + self.cfg.lookahead, len(refs) - 1)], grip)
        for _ in range(self.cfg.max_settle_steps if settle else 0):
            dp, dr = self.error(refs[-1])
            if dp < self.cfg.settle_tolerance and dr < 0.03:
                break
            self.act(refs[-1], grip)

    def hold(self, X, grip, steps):
        for _ in range(steps):
            self.act(X, grip)

    def arrays(self):
        out = {k: np.asarray(v) for k, v in self.data.items()}
        out["actions"] = out["actions"].astype(np.float64)
        return out


def sample_scene(env, rng, base, boxes, stack, cfg):
    """Uniform x-y placement of every free object; rejects overlaps and unstable scenes."""
    m = env.model
    adr = {n: 1 + m.jnt_qposadr[m.joint(n + "_joint0").id] for n in FREE_OBJECTS}

    def pairs():
        query = env.plant.get_geometry_query_input_port().Eval(env.plant_context)
        inspector = query.inspector()
        name = lambda g: env.plant.GetBodyFromFrameId(inspector.GetFrameId(g)).name()  # noqa: E731
        return {frozenset((name(p.id_A), name(p.id_B))) for p in query.ComputePointPairPenetration()}

    env.set_state(base)
    allowed = pairs()
    for attempt in range(50):
        state = base.copy()
        shift = {}
        for n in FREE_OBJECTS:
            if n not in stack:
                lo, hi = np.asarray(boxes[n][0]), np.asarray(boxes[n][1])
                xy = rng.uniform(lo - cfg.region_margin, hi + cfg.region_margin)
                shift[n] = xy - base[adr[n] : adr[n] + 2]
        for n in stack:
            root = n
            while root in stack:
                root = stack[root]
            shift[n] = shift[root]
        for n, d in shift.items():
            state[adr[n] : adr[n] + 2] += d
        state[1 + m.nq :] = 0.0
        env.set_state(state)
        if not pairs() <= allowed:
            continue
        env.reset(state)
        for k in range(cfg.scene_settle_steps):  # render only the final frame
            obs = env.step(np.r_[np.zeros(6), OPEN], observe=k == cfg.scene_settle_steps - 1)[0]
        # Drake's contact differs from MuJoCo's (the bowl sinks ~2 cm into the
        # ramekin), so require rest and no falls rather than an unchanged pose.
        settled = mujoco_state(env)
        stable = True
        for n in FREE_OBJECTS:
            delta = settled[adr[n] : adr[n] + 3] - state[adr[n] : adr[n] + 3]
            dof = 1 + m.nq + m.jnt_dofadr[m.joint(n + "_joint0").id]
            speed = np.linalg.norm(settled[dof : dof + 3])
            stable &= np.linalg.norm(delta[:2]) < 0.015 and abs(delta[2]) < 0.03 and speed < 0.01
        if stable:
            return obs, attempt
    raise PlanningError("no valid scene in 50 samples")


def grasp_sides(X_WB):
    toward = 1.0 if X_WB.translation()[1] < GRASP_SPLIT_Y else -1.0
    return [np.array(v, float) for v in ((0, toward, 0), (0, -toward, 0), (-1, 0, 0), (1, 0, 0))]


def run_expert(env, rec, cfg, rng):
    """Pick the bowl and place it on the plate; returns a short description."""
    P = Planner(env, cfg, rng)
    free = dict(margin=cfg.clearance)
    X_WB, X_WP = P.pose(BOWL), P.pose(PLATE)
    R_start = P.fk(P.q_now).rotation().matrix()

    # Grasp: first side (in a fixed preference order) with a clear approach.
    for side in grasp_sides(X_WB):
        R = P.top_down(side, R_start)
        p = X_WB.translation() + cfg.grasp_radius * side + [0, 0, cfg.grasp_height]
        X_grasp = RigidTransform(R, p)
        X_pre = raised(X_grasp, cfg.approach)
        q_pre = P.ik(X_pre, P.q_now)
        if q_pre is None or not P.clear(q_pre, **free):
            continue
        descent = P.straight(q_pre, X_grasp, ignore={BOWL}, margin=cfg.fine_clearance)
        if descent is not None:
            break
    else:
        raise PlanningError("no feasible grasp")
    approach = P.connect(P.q_now, q_pre, **free)

    rec.follow(P.timed(approach, cfg.speed), OPEN)
    rec.follow(P.timed(descent, cfg.fine_speed), OPEN)
    rec.hold(X_grasp, CLOSE, cfg.close_steps)

    # Lift straight up; the bowl leaves its support, so only the arm is checked.
    P.sync()
    bowl_z = P.pose(BOWL).translation()[2]
    X_lift = raised(P.fk(P.q_now), cfg.lift)
    lift = P.straight(P.q_now, X_lift, ignore={BOWL}, margin=cfg.fine_clearance)
    if lift is None:
        raise PlanningError("no straight lift")
    rec.follow(P.timed(lift, cfg.fine_speed), CLOSE)
    P.sync()
    if P.pose(BOWL).translation()[2] < bowl_z + 0.5 * cfg.lift:
        raise PlanningError("bowl not lifted")

    # Place: carry with the measured bowl-in-gripper pose, so slip is corrected.
    X_WG = P.fk(P.q_now)
    X_WB = P.pose(BOWL)
    X_GB = X_WG.inverse() @ X_WB
    target = np.r_[X_WP.translation()[:2], X_WP.translation()[2] + cfg.place_height]
    X_place = RigidTransform(X_WG.rotation(), X_WG.translation() + target - X_WB.translation())
    X_preplace = raised(X_place, cfg.preplace)
    q_preplace = P.ik(X_preplace, P.q_now)
    if q_preplace is None:
        raise PlanningError("no IK for pre-place")
    carry = P.connect(P.q_now, q_preplace, attached=X_GB, **free)
    lower = P.straight(
        q_preplace, X_place, attached=X_GB, ignore={PLATE}, margin=cfg.fine_clearance
    )
    if lower is None:
        raise PlanningError("no straight place descent")
    rec.follow(P.timed(carry, cfg.speed), CLOSE)
    rec.follow(P.timed(lower, cfg.fine_speed), CLOSE)
    rec.hold(X_place, OPEN, cfg.open_steps)

    P.sync()
    retreat = P.straight(
        P.q_now, raised(P.fk(P.q_now), cfg.retreat), ignore={BOWL}, margin=cfg.fine_clearance
    )
    if retreat is None:
        raise PlanningError("no straight retreat")
    rec.follow(P.timed(retreat, cfg.fine_speed), OPEN, settle=False)
    return {
        "grasp_side": side.tolist(),
        "approach_waypoints": len(approach),
        "carry_waypoints": len(carry),
    }


def generate_episode(job):
    """Worker: one attempt. Returns arrays on success, else the failure reason."""
    task_id, attempt, seed, cfg, data_dir, libero_root, boxes, video_dir = job
    cfg = ExpertConfig(**cfg)
    rng = np.random.default_rng([seed, task_id, attempt])
    source = int(rng.integers(50))
    start = time.time()
    row = {
        "task_id": task_id,
        "attempt": attempt,
        "seed": seed,
        "source_demo": source,
        "success": False,
    }
    env = LiberoEnv(
        data_dir,
        task_id,
        f"demo_{source}",
        libero_root,
        render=True,
        video=video_dir is not None,
        eye_in_hand=True,
    )
    try:
        with h5py.File(env.path, "r") as f:
            group = f[f"data/demo_{source}"]
            base = group["states"][0]
            model_file = group.attrs["model_file"]
        obs, scene_rejections = sample_scene(
            env, rng, base, boxes, stacking(task_id, libero_root), cfg
        )
        rec = Recorder(env, obs, cfg, video=video_dir is not None)
        row["scene_rejections"] = scene_rejections
        row.update(run_expert(env, rec, cfg, rng))
        # Let the released bowl come to rest on the plate.
        X_end = env.tool.CalcPoseInWorld(env.plant_context)
        rec.hold(X_end, OPEN, 3)
        row["success"] = bool(env.success())
        row["reason"] = None if row["success"] else "bowl not on plate at the end"
        row["steps"] = rec.steps
        if video_dir is not None and rec.frames:
            path = Path(video_dir) / f"task_{task_id:02d}_attempt_{attempt:05d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(path, rec.frames, fps=20, codec="libx264", macro_block_size=1)
        arrays = rec.arrays() if row["success"] else None
    except PlanningError as e:
        row["reason"], arrays, model_file = str(e), None, None
    except Exception as e:  # keep the pool alive; report the failure
        row["reason"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc(limit=4)
        arrays, model_file = None, None
    finally:
        env.close()
    row["seconds"] = round(time.time() - start, 1)
    return row, arrays, model_file


def write_episode(f, key, row, arrays, model_file):
    g = f.create_group(f"data/{key}")
    for name in ("actions", "states", "robot_states"):
        g.create_dataset(name, data=arrays[name])
    n = len(arrays["actions"])
    done = np.zeros(n, np.uint8)
    done[-1] = 1
    g.create_dataset("rewards", data=done)
    g.create_dataset("dones", data=done)
    obs = g.create_group("obs")
    for name in ("agentview_rgb", "eye_in_hand_rgb"):
        obs.create_dataset(name, data=arrays[name], compression="gzip", compression_opts=4)
    for name in Recorder.PROPRIO:
        obs.create_dataset(name, data=arrays[name])
    g.attrs["model_file"] = model_file
    g.attrs["init_state"] = arrays["states"][0]
    g.attrs["num_samples"] = n
    g.attrs["generator"] = json.dumps(row)


def finalize(path, task_id, episodes, cfg, seed, data_dir=DATA):
    """Keep the first `episodes` successes by attempt index, as demo_0..demo_{N-1}."""
    with h5py.File(Path(data_dir) / f"{NAMES[task_id]}_demo.hdf5", "r") as human:
        shared = {
            k: human["data"].attrs[k]
            for k in ("env_name", "env_args", "problem_info", "bddl_file_name")
        }
    with h5py.File(path, "a") as f:
        data = f["data"]
        keys = sorted(k for k in data if k.startswith("attempt_"))
        for k in keys[episodes:]:
            del data[k]
        keys = keys[:episodes]
        for i, k in enumerate(keys):
            data.move(k, f"demo_{i}")
        data.attrs.update(shared)
        data.attrs["macros_image_convention"] = "opengl"
        data.attrs["num_demos"] = len(keys)
        data.attrs["total"] = int(sum(data[f"demo_{i}/actions"].shape[0] for i in range(len(keys))))
        data.attrs["tag"] = GENERATOR_TAG
        data.attrs["generator_config"] = json.dumps({"seed": seed, **asdict(cfg)})
    return len(keys)


def generate(
    task_ids=DEFAULT_TASKS,
    episodes=50,
    output="data/generated/libero_spatial",
    workers=4,
    seed=0,
    cfg=None,
    data_dir=DATA,
    libero_root=ASSETS,
    videos=0,
    max_attempts=None,
):
    """Generate `episodes` successful demonstrations per task, in parallel workers."""
    cfg = cfg or ExpertConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    video_dir = output.parent / "videos"
    max_attempts = max_attempts or 4 * episodes + 10
    boxes, files, need, next_attempt, done = {}, {}, {}, {}, {}
    for t in task_ids:
        source = Path(data_dir) / f"{NAMES[t]}_demo.hdf5"
        model, _ = load_recording_model(source, "demo_0", libero_root)
        boxes[t] = placement_boxes(t, model, libero_root)
        files[t] = output / f"{NAMES[t]}_demo.hdf5.partial"
        files[t].unlink(missing_ok=True)
        with h5py.File(files[t], "w") as f:
            f.create_group("data")
        need[t], next_attempt[t], done[t] = episodes, 0, []
    log = (output / "generation_log.jsonl").open("a")
    pending = {}
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:

        def submit():
            while len(pending) < workers:
                # Optimistically count in-flight attempts as successes; failures resubmit.
                open_tasks = [
                    t
                    for t in task_ids
                    if sum(r["success"] for r in done[t])
                    + sum(1 for (u, _) in pending.values() if u == t)
                    < need[t]
                    and next_attempt[t] < max_attempts
                ]
                if not open_tasks:
                    return
                t = min(open_tasks, key=lambda u: next_attempt[u])
                a = next_attempt[t]
                next_attempt[t] += 1
                job = (
                    t,
                    a,
                    seed,
                    asdict(cfg),
                    str(data_dir),
                    str(libero_root),
                    boxes[t],
                    str(video_dir) if a < videos else None,
                )
                pending[pool.submit(generate_episode, job)] = (t, a)

        submit()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                t, a = pending.pop(future)
                row, arrays, model_file = future.result()
                done[t].append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if row["success"]:
                    with h5py.File(files[t], "a") as f:
                        write_episode(f, f"attempt_{a:05d}", row, arrays, model_file)
                ok = sum(r["success"] for r in done[t])
                status = "ok" if row["success"] else f"failed: {row.get('reason')}"
                print(
                    f"task {t} attempt {a}: {status} "
                    f"({row.get('steps', '-')} steps, {row['seconds']} s) [{ok}/{need[t]}]",
                    flush=True,
                )
            submit()
    log.close()
    summary = {}
    for t in task_ids:
        kept = finalize(files[t], t, episodes, cfg, seed, data_dir)
        final = output / f"{NAMES[t]}_demo.hdf5"
        files[t].replace(final)
        attempts = len(done[t])
        successes = sum(r["success"] for r in done[t])
        summary[t] = {"file": str(final), "demos": kept, "attempts": attempts, "successes": successes}
        print(f"task {t}: wrote {kept} demos to {final} ({successes}/{attempts} attempts succeeded)")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", type=int, choices=range(10), default=list(DEFAULT_TASKS))
    parser.add_argument("--episodes", type=int, default=50, help="Successful demos per task")
    parser.add_argument("--output", type=Path, default=Path("data/generated/libero_spatial"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--videos", type=int, default=0, help="Save videos of the first N attempts per task")
    parser.add_argument(
        "--region-margin", type=float, default=0.0, help="Widen placement regions by this many metres"
    )
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1:
        parser.error("episodes and workers must be positive")
    generate(
        args.tasks,
        args.episodes,
        args.output,
        args.workers,
        args.seed,
        ExpertConfig(region_margin=args.region_margin),
        args.data,
        args.assets,
        args.videos,
    )


if __name__ == "__main__":
    main()
