"""Original LIBERO scenes, Panda OSC, and human-action replay in Drake.

MuJoCo compiles MJCF and converts coordinates; Drake integrates all rollouts.
The 0.25 ms physics step resolves point contact without changing the colliders.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BasicVector,
    Box,
    CameraInfo,
    Capsule,
    ClippingRange,
    ColorRenderCamera,
    CoulombFriction,
    Cylinder,
    DepthRange,
    DepthRenderCamera,
    DiagramBuilder,
    Ellipsoid,
    FixedOffsetFrame,
    GeometryInstance,
    HalfSpace,
    IllustrationProperties,
    JacobianWrtVariable,
    LeafSystem,
    MakeRenderEngineVtk,
    Mesh,
    PerceptionProperties,
    PrismaticJoint,
    Quaternion,
    QuaternionFloatingJoint,
    RenderCameraCore,
    RenderEngineVtkParams,
    RevoluteJoint,
    Rgba,
    RgbdSensor,
    RigidTransform,
    RotationMatrix,
    Simulator,
    SpatialInertia,
    SpatialVelocity,
    Sphere,
    UnitInertia,
)
from scipy.spatial.transform import Rotation

DATA = Path("data/LIBERO/libero_spatial")
ASSETS = Path("external/LIBERO")

LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
DATASET_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
DATASET_REPO = "yifengzhu-hf/LIBERO-datasets"
NAMES = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
)


def transform(position, quaternion):
    return RigidTransform(Quaternion(np.asarray(quaternion)), np.asarray(position))


def remap_xml(xml, libero_root):
    root = ET.fromstring(xml)
    asset_root = Path(libero_root).resolve() / "libero/libero/assets"
    spec = importlib.util.find_spec("robosuite")
    if spec is None:
        raise ImportError("Install the project's LIBERO dependencies first")
    robot_root = Path(spec.origin).parent / "models/assets"
    for node in root.findall(".//*[@file]"):
        value = node.get("file")
        if "/robosuite/models/assets/" in value:
            path = robot_root / value.split("/robosuite/models/assets/", 1)[1]
        elif "/assets/" in value:
            path = asset_root / value.split("/assets/", 1)[1]
        else:
            path = Path(value)
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing original LIBERO asset: {path}")
        node.set("file", str(path))
    return ET.tostring(root, encoding="unicode")


def load_recording_model(path, demo, libero_root):
    with h5py.File(path, "r") as f:
        xml = f[f"data/{demo}"].attrs["model_file"]
    xml = remap_xml(xml, libero_root)
    return mujoco.MjModel.from_xml_string(xml), xml


def export_mesh(model, mesh_id, cache):
    """Use compiled vertices and geom transforms to preserve MuJoCo's mesh centering."""
    va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    verts = model.mesh_vert[va : va + vn]
    faces = model.mesh_face[fa : fa + fn]
    ta, tn = model.mesh_texcoordadr[mesh_id], model.mesh_texcoordnum[mesh_id]
    uv = model.mesh_texcoord[ta : ta + tn] if ta >= 0 else np.empty((0, 2))
    uv_faces = model.mesh_facetexcoord[fa : fa + fn]
    na, nn = model.mesh_normaladr[mesh_id], model.mesh_normalnum[mesh_id]
    normals = model.mesh_normal[na : na + nn]
    normal_faces = model.mesh_facenormal[fa : fa + fn]
    digest = hashlib.sha256(
        b"v2"
        + verts.tobytes()
        + faces.tobytes()
        + uv.tobytes()
        + uv_faces.tobytes()
        + normals.tobytes()
        + normal_faces.tobytes()
    ).hexdigest()
    path = Path(cache).resolve() / (digest + ".obj")
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["v " + " ".join(f"{float(x):.9g}" for x in v) for v in verts]
    if len(uv):
        lines += ["vt " + " ".join(f"{float(x):.9g}" for x in v) for v in uv]
    if not len(normals):
        triangles = verts[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
        normal_faces = np.repeat(np.arange(len(faces))[:, None], 3, axis=1)
    lines += ["vn " + " ".join(f"{float(x):.9g}" for x in v) for v in normals]
    for i, face in enumerate(faces):
        if len(uv) and np.all(uv_faces[i] >= 0):
            indices = [
                f"{v + 1}/{t + 1}/{n + 1}" for v, t, n in zip(face, uv_faces[i], normal_faces[i])
            ]
        else:
            indices = [f"{v + 1}//{n + 1}" for v, n in zip(face, normal_faces[i])]
        lines.append("f " + " ".join(indices))
    path.write_text("\n".join(lines) + "\n")
    return path


def import_model(plant, model, xml, cache, visuals=True):
    instance = plant.AddModelInstance("libero")
    bodies = {0: plant.world_body()}
    joints = {}
    for i in range(1, model.nbody):
        mass = model.body_mass[i]
        if mass > 0:
            R = RotationMatrix(Quaternion(model.body_iquat[i])).matrix()
            inertia = R @ np.diag(model.body_inertia[i] / mass) @ R.T
            unit = UnitInertia(
                inertia[0, 0],
                inertia[1, 1],
                inertia[2, 2],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 2],
            )
            spatial = SpatialInertia.MakeFromCentralInertia(mass, model.body_ipos[i], mass * unit)
        else:
            spatial = SpatialInertia.Zero()
        body = plant.AddRigidBody(model.body(i).name, instance, spatial)
        bodies[i] = body
        parent = bodies[int(model.body_parentid[i])]
        pose = transform(model.body_pos[i], model.body_quat[i])
        n = int(model.body_jntnum[i])
        if n == 0:
            plant.WeldFrames(parent.body_frame(), body.body_frame(), pose)
        else:
            if n != 1:
                raise ValueError(f"Multiple joints per body unsupported: {body.name()}")
            j = int(model.body_jntadr[i])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                # Automatic floating joints would silently discard MJCF damping.
                d = model.dof_damping[model.jnt_dofadr[j] : model.jnt_dofadr[j] + 6]
                if not (np.all(d[:3] == d[0]) and np.all(d[3:] == d[3])):
                    raise ValueError("Floating-joint damping must be isotropic")
                plant.AddJoint(
                    QuaternionFloatingJoint(
                        model.joint(j).name,
                        parent.body_frame(),
                        body.body_frame(),
                        angular_damping=float(d[3]),
                        translational_damping=float(d[0]),
                    )
                )
                continue
            if model.qpos0[model.jnt_qposadr[j]] != 0:
                raise ValueError("Nonzero joint reference requires explicit conversion")
            joint_pose = RigidTransform(model.jnt_pos[j])
            parent_frame = plant.AddFrame(
                FixedOffsetFrame(
                    f"{model.joint(j).name}_parent", parent.body_frame(), pose @ joint_pose
                )
            )
            child_frame = plant.AddFrame(
                FixedOffsetFrame(f"{model.joint(j).name}_child", body.body_frame(), joint_pose)
            )
            lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-np.inf, np.inf)
            damping = float(model.dof_damping[model.jnt_dofadr[j]])
            cls = (
                RevoluteJoint
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                else PrismaticJoint
            )
            joint = plant.AddJoint(
                cls(
                    model.joint(j).name,
                    parent_frame,
                    child_frame,
                    model.jnt_axis[j],
                    lo,
                    hi,
                    damping,
                )
            )
            joints[j] = joint
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        limit = max(abs(model.actuator_ctrlrange[a])) if a < 7 else 20.0
        actuator = plant.AddJointActuator(model.actuator(a).name, joints[j], limit)
        actuator.set_default_rotor_inertia(float(model.dof_armature[model.jnt_dofadr[j]]))
    # Keep exact local site frames, including the robosuite grip site.
    for s in range(model.nsite):
        plant.AddFrame(
            FixedOffsetFrame(
                "site/" + model.site(s).name,
                bodies[int(model.site_bodyid[s])].body_frame(),
                transform(model.site_pos[s], model.site_quat[s]),
            )
        )
    texture_paths = {
        n.get("name"): n.get("file") for n in ET.fromstring(xml).findall("asset/texture")
    }
    for g in range(model.ngeom):
        kind = model.geom_type[g]
        size = model.geom_size[g]
        shape = None
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            shape = Mesh(str(export_mesh(model, int(model.geom_dataid[g]), cache)))
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            shape = Box(*(size * 2))
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            shape = Sphere(size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape = Cylinder(size[0], 2 * size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            shape = Capsule(size[0], 2 * size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            shape = Ellipsoid(*size)
        elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
            shape = HalfSpace()
        if shape is None:
            raise ValueError(f"Unsupported geometry type {kind}")
        body = bodies[int(model.geom_bodyid[g])]
        pose = transform(model.geom_pos[g], model.geom_quat[g])
        name = model.geom(g).name or f"geom_{g}"
        collision = bool(model.geom_contype[g] or model.geom_conaffinity[g])
        if collision:
            mu = float(model.geom_friction[g, 0])
            plant.RegisterCollisionGeometry(body, pose, shape, name, CoulombFriction(mu, mu))
        # Robosuite's standard camera mode displays geometry group 1 only.
        if visuals and model.geom_group[g] == 1:
            rgba = model.geom_rgba[g].copy()
            mat = int(model.geom_matid[g])
            props = IllustrationProperties()
            if mat >= 0:
                rgba = model.mat_rgba[mat].copy()
                tex_ids = model.mat_texid[mat]
                tex = int(tex_ids[1])  # mjTEXROLE_RGB
                if tex >= 0:
                    path = texture_paths.get(model.tex(tex).name)
                    if path:
                        props.AddProperty("phong", "diffuse_map", path)
            rgba[3] = max(0.0, min(1.0, rgba[3]))
            props.AddProperty("phong", "diffuse", Rgba(*rgba))
            geometry = GeometryInstance(pose, shape, name)
            geometry.set_illustration_properties(props)
            perception = PerceptionProperties()
            perception.AddProperty("phong", "diffuse", Rgba(*rgba))
            if props.HasProperty("phong", "diffuse_map"):
                perception.AddProperty(
                    "phong", "diffuse_map", props.GetProperty("phong", "diffuse_map")
                )
            geometry.set_perception_properties(perception)
            plant.RegisterVisualGeometry(body, geometry)
    return instance, bodies, joints


class JointFriction(LeafSystem):
    """Smooth Coulomb friction for MuJoCo's dof_frictionloss parameters."""

    def __init__(self, plant, model, joints):
        super().__init__()
        self.plant = plant
        self.loss = np.zeros(plant.num_velocities())
        for i, joint in joints.items():
            self.loss[joint.velocity_start()] = model.dof_frictionloss[model.jnt_dofadr[i]]
        self.input = self.DeclareVectorInputPort("state", BasicVector(plant.num_multibody_states()))
        self.DeclareVectorOutputPort("friction", BasicVector(plant.num_velocities()), self.calc)

    def calc(self, context, output):
        v = self.input.Eval(context)[self.plant.num_positions() :]
        output.SetFromVector(-self.loss * np.tanh(v / 0.001))


class OscController(LeafSystem):
    def __init__(self, plant, model, joints):
        super().__init__()
        self.plant = plant
        self.model = model
        self.joints = joints
        self.kin = plant.CreateDefaultContext()
        self.arm = [joints[i] for i in range(7)]
        self.qi = [j.position_start() for j in self.arm]
        self.vi = [j.velocity_start() for j in self.arm]
        self.fingers = [joints[i] for i in (7, 8)]
        self.tool = plant.GetFrameByName("site/gripper0_grip_site")
        self.state = self.DeclareVectorInputPort("state", BasicVector(plant.num_multibody_states()))
        self.goal = self.DeclareVectorInputPort("goal", BasicVector(13))
        self.DeclareVectorOutputPort("torque", BasicVector(9), self.calc)

    def calc(self, context, output):
        state = self.state.Eval(context)
        self.plant.SetPositionsAndVelocities(self.kin, state)
        goal = self.goal.Eval(context)
        desired_pos, desired_rot = goal[:3], goal[3:12].reshape(3, 3)
        grip = goal[12]
        initial = self.initial
        pose = self.tool.CalcPoseInWorld(self.kin)
        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            self.kin,
            JacobianWrtVariable.kV,
            self.tool,
            np.zeros(3),
            self.plant.world_frame(),
            self.plant.world_frame(),
        )[:, self.vi]
        J = np.vstack([J_spatial[3:], J_spatial[:3]])
        mass = self.plant.CalcMassMatrix(self.kin)[np.ix_(self.vi, self.vi)]
        inverse_mass = np.linalg.inv(mass)
        full_lambda = np.linalg.pinv(J @ inverse_mass @ J.T)
        pos_lambda = np.linalg.pinv(J[:3] @ inverse_mass @ J[:3].T)
        rot_lambda = np.linalg.pinv(J[3:] @ inverse_mass @ J[3:].T)
        q = self.plant.GetPositions(self.kin)[self.qi]
        v = self.plant.GetVelocities(self.kin)[self.vi]
        current_rot = pose.rotation().matrix()
        orientation_error = 0.5 * sum(
            np.cross(current_rot[:, i], desired_rot[:, i]) for i in range(3)
        )
        velocity = J @ v
        acceleration = (
            150 * np.r_[desired_pos - pose.translation(), orientation_error]
            - 2 * np.sqrt(150) * velocity
        )
        wrench = np.r_[pos_lambda @ acceleration[:3], rot_lambda @ acceleration[3:]]
        nullspace = np.eye(7) - inverse_mass @ J.T @ full_lambda @ J
        bias = self.plant.CalcBiasTerm(self.kin) - self.plant.CalcGravityGeneralizedForces(self.kin)
        torque = J.T @ wrench + bias[self.vi]
        torque += nullspace.T @ mass @ (10 * (initial - q) - 2 * np.sqrt(10) * v)
        # Robosuite's finger actuators are 1000 N/m position servos; joint damping
        # (100 N s/m) and rotor inertia (1) are already in the imported plant.
        finger_torques = []
        for sign, joint in zip((1, -1), self.fingers):
            target = sign * (0.02 * (1 + grip))
            pos = self.plant.GetPositions(self.kin)[joint.position_start()]
            finger_torques.append(np.clip(1000 * (target - pos), -20, 20))
        output.SetFromVector(
            np.r_[np.clip(torque, [-80] * 5 + [-12] * 2, [80] * 5 + [12] * 2), finger_torques]
        )


class LiberoEnv:
    def __init__(
        self,
        dataset_dir="data/LIBERO/libero_spatial",
        task_id=0,
        demo="demo_0",
        libero_root="external/LIBERO",
        time_step=0.00025,
        image_size=128,
        render=True,
        video=False,
    ):
        render = render or video
        self.task_id = task_id
        self.task = NAMES[task_id]
        self.path = Path(dataset_dir) / (self.task + "_demo.hdf5")
        self.demo = demo
        self.model, self.xml = load_recording_model(self.path, demo, libero_root)
        self.mj_data = mujoco.MjData(self.model)
        self.time_step, self.image_size = time_step, image_size
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
        self.instance, self.bodies, self.joints = import_model(
            self.plant,
            self.model,
            self.xml,
            Path(dataset_dir).parent / "drake_meshes",
            visuals=render,
        )
        self.plant.Finalize()
        self.tool = self.plant.GetFrameByName("site/gripper0_grip_site")
        self.controller = builder.AddSystem(OscController(self.plant, self.model, self.joints))
        builder.Connect(self.plant.get_state_output_port(), self.controller.GetInputPort("state"))
        builder.Connect(self.controller.get_output_port(), self.plant.get_actuation_input_port())
        self.friction = builder.AddSystem(JointFriction(self.plant, self.model, self.joints))
        builder.Connect(self.plant.get_state_output_port(), self.friction.get_input_port())
        builder.Connect(
            self.friction.get_output_port(), self.plant.get_applied_generalized_force_input_port()
        )
        builder.ExportInput(self.controller.GetInputPort("goal"), "goal")
        self.cameras = {}
        if render:
            self.scene_graph.AddRenderer("libero", MakeRenderEngineVtk(RenderEngineVtkParams()))
            resolutions = {"agentview_rgb": image_size}
            if video:
                resolutions["video_rgb"] = 1024
            for name, size in resolutions.items():
                cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
                bid = int(self.model.cam_bodyid[cid])
                frame = (
                    self.scene_graph.world_frame_id()
                    if bid == 0
                    else self.plant.GetBodyFrameIdOrThrow(self.bodies[bid].index())
                )
                # MuJoCo cameras look down -Z with +Y up; Drake optical frames use +Z and -Y.
                pose = transform(
                    self.model.cam_pos[cid], self.model.cam_quat[cid]
                ) @ RigidTransform(RotationMatrix.MakeXRotation(np.pi))
                core = RenderCameraCore(
                    "libero",
                    CameraInfo(size, size, np.deg2rad(self.model.cam_fovy[cid])),
                    ClippingRange(0.001, 10),
                    RigidTransform(),
                )
                sensor = builder.AddSystem(
                    RgbdSensor(
                        frame,
                        pose,
                        ColorRenderCamera(core, False),
                        DepthRenderCamera(core, DepthRange(0.001, 10)),
                    )
                )
                builder.Connect(
                    self.scene_graph.get_query_output_port(), sensor.query_object_input_port()
                )
                self.cameras[name] = sensor
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.simulator = None

    def close(self):
        """Release Python controller references before destroying the owning diagram."""
        self.simulator = None
        self.context = None
        self.controller.kin = None
        self.controller.plant = None
        self.controller.joints = None
        self.controller.arm = []
        self.controller.fingers = []
        self.controller.tool = None
        self.controller.state = self.controller.goal = None
        self.friction.plant = self.friction.input = None
        self.cameras.clear()
        self.bodies.clear()
        self.joints.clear()
        self.tool = self.plant = self.scene_graph = None
        self.controller = self.friction = self.diagram = None
        self.mj_data = self.model = None
        gc.collect()

    @property
    def plant_context(self):
        return self.plant.GetMyMutableContextFromRoot(self.context)

    def set_state(self, flattened):
        state = np.asarray(flattened)
        m, d = self.model, self.mj_data
        if len(state) != 1 + m.nq + m.nv:
            raise ValueError("Recording state size does not match its embedded XML")
        d.time = state[0]
        d.qpos[:] = state[1 : 1 + m.nq]
        d.qvel[:] = state[1 + m.nq :]
        mujoco.mj_forward(m, d)  # Kinematics only; no MuJoCo integration.
        q = self.plant.GetPositions(self.plant_context).copy()
        v = self.plant.GetVelocities(self.plant_context).copy()
        for j, joint in self.joints.items():
            q[joint.position_start()] = d.qpos[m.jnt_qposadr[j]]
            v[joint.velocity_start()] = d.qvel[m.jnt_dofadr[j]]
        self.plant.SetPositions(self.plant_context, q)
        self.plant.SetVelocities(self.plant_context, v)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                bid = int(m.jnt_bodyid[j])
                body = self.bodies[bid]
                self.plant.SetFreeBodyPose(
                    self.plant_context, body, transform(d.xpos[bid], d.xquat[bid])
                )
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, bid, velocity, 0)
                self.plant.SetFreeBodySpatialVelocity(
                    self.plant_context, body, SpatialVelocity(velocity[:3], velocity[3:])
                )

    def reset(self, state=None):
        self.context = self.diagram.CreateDefaultContext()
        if state is None:
            with h5py.File(self.path, "r") as f:
                state = f[f"data/{self.demo}/states"][0]
        self.set_state(state)
        self.steps = 0
        self.grip = 1.0  # Dataset creation opens the gripper during its discarded prefix.
        self.initial_height = self.body_position("akita_black_bowl_1_main")[2]
        self.lifted = False
        # Robosuite's nominal mounted-Panda posture is the nullspace reference.
        self.controller.initial = np.array(
            [0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.22675220, np.pi / 4]
        )
        pose = self.tool.CalcPoseInWorld(self.plant_context)
        self.goal_pos, self.goal_rot = pose.translation(), pose.rotation().matrix()
        self._fix_goal()
        self.simulator = Simulator(self.diagram, self.context)
        self.simulator.Initialize()
        return self.observe()

    def _fix_goal(self):
        self.diagram.GetInputPort("goal").FixValue(
            self.context, np.r_[self.goal_pos, self.goal_rot.ravel(), self.grip]
        )

    def body_position(self, name):
        body = self.plant.GetBodyByName(name)
        return self.plant.EvalBodyPoseInWorld(self.plant_context, body).translation().copy()

    def success(self):
        bowl = self.body_position("akita_black_bowl_1_main")
        plate = self.body_position("plate_1_main")
        if bowl[2] < plate[2] or np.linalg.norm(bowl[:2] - plate[:2]) >= 0.03:
            return False
        contacts = self.plant.get_contact_results_output_port().Eval(self.plant_context)
        bowl_ids = {self.plant.GetBodyByName("akita_black_bowl_1_main").index()}
        plate_id = self.plant.GetBodyByName("plate_1_main").index()
        for i in range(contacts.num_point_pair_contacts()):
            contact = contacts.point_pair_contact_info(i)
            pair = {contact.bodyA_index(), contact.bodyB_index()}
            if plate_id in pair and pair & bowl_ids:
                return True
        return False

    def step(self, action, observe=True):
        a = np.asarray(action)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError("Expected seven finite LIBERO OSC actions")
        a = np.clip(a, -1, 1)
        pose = self.tool.CalcPoseInWorld(self.plant_context)
        self.goal_pos = pose.translation() + 0.05 * a[:3]
        # Robosuite retains the previous orientation goal on an exactly zero delta.
        if np.any(a[3:6] != 0):
            self.goal_rot = (
                Rotation.from_rotvec(0.5 * a[3:6]).as_matrix() @ pose.rotation().matrix()
            )
        # Keep robosuite's 2 ms gripper updates (25/action), independently of
        # Drake's finer physics step needed to resolve flat point contacts.
        control_step = float(self.model.opt.timestep)
        for _ in range(round(0.05 / control_step)):
            self.grip = np.clip(self.grip - 0.01 * np.sign(a[6]), -1, 1)
            self._fix_goal()
            self.simulator.AdvanceTo(self.context.get_time() + control_step)
        self.steps += 1
        self.lifted |= bool(
            self.body_position("akita_black_bowl_1_main")[2] > self.initial_height + 0.04
        )
        success = self.success()
        info = {"success": success, "lifted": self.lifted, "steps": self.steps}
        return self.observe() if observe else None, float(success), success, info

    def observe(self):
        obs = {}
        for name, sensor in self.cameras.items():
            context = sensor.GetMyContextFromRoot(self.context)
            obs[name] = sensor.color_image_output_port().Eval(context).data[:, :, :3].copy()
        return obs


def download(data_dir=DATA, libero_root=ASSETS):
    from huggingface_hub import snapshot_download

    source = Path(libero_root)
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/Lifelong-Robot-Learning/LIBERO.git", str(source)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", LIBERO_COMMIT], check=True
        )
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != LIBERO_COMMIT:
        raise ValueError(f"Expected LIBERO assets at {LIBERO_COMMIT}; found {actual}")
    # snapshot_download preserves the official libero_spatial/ directory.
    if Path(data_dir).name != "libero_spatial":
        raise ValueError("Dataset directory must end in libero_spatial")
    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=[f"libero_spatial/{name}_demo.hdf5" for name in NAMES],
        local_dir=Path(data_dir).parent,
        max_workers=3,
    )
    for name in NAMES:
        with h5py.File(Path(data_dir) / (name + "_demo.hdf5"), "r") as f:
            if len(f["data"]) != 50:
                raise ValueError(f"Expected 50 original human demonstrations for {name}")
    print("Ready: ten original tasks and 500 human demonstrations")


def load_initial_states(libero_root, task_id):
    import torch

    path = (
        Path(libero_root)
        / "libero/libero/init_files/libero_spatial"
        / (NAMES[task_id] + ".pruned_init")
    )
    with torch.serialization.safe_globals(
        [
            (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
            np.ndarray,
            np.dtype,
            np.dtypes.Float64DType,
            np.dtypes.Float32DType,
        ]
    ):
        return np.asarray(torch.load(path, weights_only=True))


def replay(
    data_dir=DATA,
    libero_root=ASSETS,
    task_ids=tuple(range(10)),
    episodes=1,
    output="runs/replay",
    video=False,
):
    """Replay every recorded action, without resetting objects between steps."""
    task_ids = tuple(task_ids)
    if not task_ids or any(t not in range(10) for t in task_ids) or not 1 <= episodes <= 50:
        raise ValueError("Choose task IDs 0–9 and 1–50 human episodes per task")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_id in task_ids:
        for index in range(episodes):
            # Each recording embeds its own fixture placement in the MJCF.
            env = LiberoEnv(
                data_dir, task_id, f"demo_{index}", libero_root, render=video, video=video
            )
            writer = None
            try:
                obs = env.reset()
                with h5py.File(env.path, "r") as f:
                    actions = f[f"data/{env.demo}/actions"][:]
                if video:
                    writer = imageio.get_writer(
                        output / f"task_{task_id:02d}_{index:02d}.mp4",
                        fps=20,
                        codec="libx264",
                        macro_block_size=1,
                    )
                success = False
                for action in actions:
                    if writer:
                        writer.append_data(obs["video_rgb"])
                    obs, _, done, info = env.step(action, observe=video)
                    success |= done
                if writer:
                    writer.append_data(obs["video_rgb"])
                row = {
                    "task_id": task_id,
                    "episode": index,
                    "success": success,
                    "final_success": info["success"],
                    "steps": len(actions),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                report = {
                    "engine": "Drake",
                    "physics_time_step": env.time_step,
                    "mode": "human replay",
                    "success_rate": float(np.mean([r["success"] for r in rows])),
                    "episodes": rows,
                }
                (output / "results.json").write_text(json.dumps(report, indent=2))
            finally:
                if writer:
                    writer.close()
                env.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "replay"):
        p = sub.add_parser(command)
        p.add_argument("--data", type=Path, default=DATA)
        p.add_argument("--assets", type=Path, default=ASSETS)
        if command == "replay":
            p.add_argument(
                "--tasks", nargs="+", type=int, choices=range(10), default=list(range(10))
            )
            p.add_argument("--episodes", type=int, choices=range(1, 51), default=1)
            p.add_argument("--output", type=Path, default=Path("runs/replay"))
            p.add_argument("--video", action="store_true", help="Save native 1024x1024 video")
    args = parser.parse_args()
    if args.command == "download":
        download(args.data, args.assets)
    else:
        replay(args.data, args.assets, args.tasks, args.episodes, args.output, args.video)


if __name__ == "__main__":
    main()
