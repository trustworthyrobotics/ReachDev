import numpy as np

from drake import lcmt_iiwa_command, lcmt_iiwa_status
from pydrake.all import (
    DiagramBuilder,
    DrakeLcm,
    LcmInterfaceSystem,
    LcmSubscriberSystem,
    LcmPublisherSystem,
    IiwaStatusReceiver,
    IiwaCommandSender,
    IiwaControlMode,
    position_enabled,
    LeafSystem,
    BasicVector,
    Simulator,
    InitializeAutoDiff,  
)

def MakeLcm(builder: DiagramBuilder):
    lcm = DrakeLcm()
    builder.AddSystem(LcmInterfaceSystem(lcm))
    return lcm

def MakeSingleIiwaLcmIO(builder: DiagramBuilder, lcm, lcm_channel_suffix=""):
    """
    Exports:
      input:  "position" (7)
      output: "position_measured" (7)
    """
    control_mode = IiwaControlMode.kPositionAndTorque
    assert position_enabled(control_mode)

    # Command publisher (200 Hz in this mode) :contentReference[oaicite:3]{index=3}
    cmd_sender = builder.AddSystem(IiwaCommandSender(control_mode=control_mode))
    cmd_pub = builder.AddSystem(
        LcmPublisherSystem.Make(
            channel="IIWA_COMMAND" + lcm_channel_suffix,
            lcm_type=lcmt_iiwa_command,
            lcm=lcm,
            publish_period=0.005,
            use_cpp_serializer=True,
        )
    )
    builder.Connect(cmd_sender.get_output_port(), cmd_pub.get_input_port())
    builder.ExportInput(cmd_sender.get_position_input_port(), "position")

    # Status subscriber
    status_recv = builder.AddSystem(IiwaStatusReceiver())
    status_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel="IIWA_STATUS" + lcm_channel_suffix,
            lcm_type=lcmt_iiwa_status,
            lcm=lcm,
            use_cpp_serializer=True,
            wait_for_message_on_initialization_timeout=10,
        )
    )
    builder.Connect(status_sub.get_output_port(), status_recv.get_input_port())
    builder.ExportOutput(status_recv.get_position_measured_output_port(), "position_measured")

    return (cmd_sender, status_recv)

class JointPositionStreamer(LeafSystem):
    """
    Inputs:
      - q_measured (7)
      - q_goal     (7)
    Output:
      - q_cmd      (7)  (stateful, rate-limited)
    """
    def __init__(self, period_sec=0.005, max_delta_per_step=0.01):
        super().__init__()
        self._max_delta = float(max_delta_per_step)

        self.DeclareVectorInputPort("q_measured", 7)
        self.DeclareVectorInputPort("q_goal", 7)

        state_idx = self.DeclareDiscreteState(7)  # q_cmd
        self.DeclareStateOutputPort("q_cmd", state_idx)

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=float(period_sec),
            offset_sec=0.0,
            update=self._update,
        )

    def _update(self, context, discrete_state):
        q_meas = self.get_input_port(0).Eval(context)
        q_goal = self.get_input_port(1).Eval(context)

        # Initialize q_cmd from q_measured on first tick (optional but helpful)
        q_cmd = discrete_state.get_vector().CopyToVector()
        if not np.isfinite(q_cmd).all() or np.linalg.norm(q_cmd) < 1e-12:
            q_cmd = np.array(q_meas, dtype=float)

        delta = np.clip(q_goal - q_cmd, -self._max_delta, +self._max_delta)
        q_next = q_cmd + delta
        discrete_state.set_value(q_next)


from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import FixedOffsetFrame

class ToolPoseMetrics(LeafSystem):
    """
    Input:  q (7)
    Output: metrics (4) = [x, y, z, align_score]
      align_score = dot(tool_z_world, -world_z) in [ -1, 1 ]
    """
    def __init__(self, plant: MultibodyPlant, model_instance, tool_frame):
        super().__init__()
        self._plant = plant
        self._model = model_instance
        self._tool = tool_frame
        self._ctx = self._plant.CreateDefaultContext()

        self.DeclareVectorInputPort("q", 7)
        self.DeclareVectorOutputPort("metrics", BasicVector(4), self._calc)

    def _calc(self, context, output):
        q = self.get_input_port(0).Eval(context)
        self._plant.SetPositions(self._ctx, self._model, q)

        X_WT = self._plant.CalcRelativeTransform(self._ctx, self._plant.world_frame(), self._tool)
        p = np.array(X_WT.translation(), dtype=float)
        R = X_WT.rotation().matrix()
        tool_z_world = R[:, 2]
        align_score = float(np.dot(tool_z_world, np.array([0.0, 0.0, -1.0])))

        output.SetFromVector(np.array([p[0], p[1], p[2], align_score], dtype=float))


def BuildFkPlantWithToolFrame(
    urdf_url="package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf",
    tool_offset_z=0.10,
):
    """
    Minimal plant (no SceneGraph) for FK computations only.
    Tool frame definition matches your sim: link7 + XRotation(pi) + z offset. :contentReference[oaicite:6]{index=6}
    """
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    model_instance = parser.AddModelsFromUrl(urdf_url)[0]

    # Weld base (fixed-base IIWA) :contentReference[oaicite:7]{index=7}
    base_body = plant.GetBodyByName("base", model_instance)
    plant.WeldFrames(plant.world_frame(), base_body.body_frame(), RigidTransform())

    # Tool frame :contentReference[oaicite:8]{index=8}
    link7 = plant.GetFrameByName("iiwa_link_7", model_instance)
    X_7T = RigidTransform(
        RotationMatrix.MakeXRotation(np.pi),
        [0.0, 0.0, float(tool_offset_z)],
    )
    tool_frame = plant.AddFrame(FixedOffsetFrame("tool_frame", link7, X_7T))

    plant.Finalize()
    return plant, model_instance, tool_frame

from pydrake.all import ConstantVectorSource

def MakeIiwaRealStation(
    lcm_channel_suffix="",
    urdf_url="package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf",
    tool_offset_z=0.10,
    period_sec=0.005,
    max_delta_per_step=0.01,
):
    """
    Exports:
      input:
        - q_goal (7)  (desired joint target from planner)
      output:
        - q_measured (7)
        - tool_metrics (4) = [x,y,z,align_score]
    """
    builder = DiagramBuilder()
    lcm = MakeLcm(builder)

    # LCM IO
    MakeSingleIiwaLcmIO(builder, lcm, lcm_channel_suffix=lcm_channel_suffix)

    # Get handles to exported ports via builder’s diagram later (we wire by name below)
    # Controller
    streamer = builder.AddSystem(JointPositionStreamer(period_sec=period_sec, max_delta_per_step=max_delta_per_step))
    builder.Connect(builder.GetExportedOutputPort("position_measured"), streamer.GetInputPort("q_measured"))

    # q_goal input to station
    builder.ExportInput(streamer.GetInputPort("q_goal"), "q_goal")

    # streamer's q_cmd to IIWA command input
    builder.Connect(streamer.GetOutputPort("q_cmd"), builder.GetExportedInputPort("position"))

    # Export measured q
    builder.ExportOutput(builder.GetExportedOutputPort("position_measured"), "q_measured")

    # FK tool metrics from measured q (separate FK plant)
    fk_plant, fk_model, fk_tool = BuildFkPlantWithToolFrame(urdf_url=urdf_url, tool_offset_z=tool_offset_z)
    metrics_sys = builder.AddSystem(ToolPoseMetrics(fk_plant, fk_model, fk_tool))
    builder.Connect(builder.GetExportedOutputPort("position_measured"), metrics_sys.GetInputPort("q"))
    builder.ExportOutput(metrics_sys.GetOutputPort("metrics"), "tool_metrics")

    return builder.Build()

import numpy as np
import time

from pydrake.all import (
    LeafSystem, BasicVector,
    MultibodyPlant, Parser,
    RigidTransform, RotationMatrix,
    InverseKinematics, Solve,
)
from pydrake.multibody.tree import FixedOffsetFrame


class IiwaIkGoalManager(LeafSystem):
    """
    Event-driven IK manager:
      - runs IK at a low rate (e.g., 10-20 Hz)
      - holds last good q_goal
      - outputs q_goal for a fast streaming controller (200 Hz)
    """
    def __init__(
        self,
        urdf_url="package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf",
        tool_offset_z=0.10,
        ang_tol_deg=5.0,
        update_period_sec=0.05,      # 20 Hz IK loop
        target_change_tol=1e-4,      # meters
        seed_blend=0.0,              # optional: blend last q_goal with q_measured
    ):
        super().__init__()
        self._ang_tol_deg = float(ang_tol_deg)
        self._update_period = float(update_period_sec)
        self._target_tol = float(target_change_tol)
        self._seed_blend = float(seed_blend)

        # Build a tiny IK plant (no SceneGraph) for solving IK
        self._plant = MultibodyPlant(time_step=0.0)
        parser = Parser(self._plant)
        self._model = parser.AddModelsFromUrl(urdf_url)[0]

        base_body = self._plant.GetBodyByName("base", self._model)
        self._plant.WeldFrames(self._plant.world_frame(), base_body.body_frame(), RigidTransform())

        link7 = self._plant.GetFrameByName("iiwa_link_7", self._model)
        X_7T = RigidTransform(
            RotationMatrix.MakeXRotation(np.pi),
            [0.0, 0.0, float(tool_offset_z)],
        )
        self._tool = self._plant.AddFrame(FixedOffsetFrame("tool_frame", link7, X_7T))
        self._plant.Finalize()
        self._plant_ad = self._plant.ToAutoDiffXd()
        self._ctx_ad = self._plant_ad.CreateDefaultContext()
        self._model_ad = self._plant_ad.GetModelInstanceByName(self._model_name)
        self._tool_ad = self._plant_ad.GetFrameByName(self._tool.name(), self._model_ad)



        self._nq = self._plant.num_positions(self._model)

        # Ports
        self.DeclareVectorInputPort("target_xyz", 3)
        self.DeclareVectorInputPort("q_measured", self._nq)

        # Discrete state:
        #   [0:3]     last_target_xyz
        #   [3:3+nq]  q_goal
        #   [..]      success_flag (0/1)
        #   [..]      last_solve_time_sec
        self._idx_last_target = self.DeclareDiscreteState(3)
        self._idx_q_goal = self.DeclareDiscreteState(self._nq)
        self._idx_meta = self.DeclareDiscreteState(2)  # [success_flag, last_solve_time]

        self.DeclareStateOutputPort("q_goal", self._idx_q_goal)
        self.DeclareVectorOutputPort("ik_status", BasicVector(2), self._calc_status)

        # Periodic update (slow loop)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=self._update_period,
            offset_sec=0.0,
            update=self._update,
        )

    def _calc_status(self, context, output):
        meta = context.get_discrete_state(self._idx_meta).get_value()
        output.SetFromVector(meta)

    def _update(self, context, discrete_state):
        target = np.array(self.get_input_port(0).Eval(context), dtype=float).reshape(3)
        q_meas = np.array(self.get_input_port(1).Eval(context), dtype=float).reshape(self._nq)

        last_target = discrete_state.get_vector(self._idx_last_target).CopyToVector()
        q_goal_prev = discrete_state.get_vector(self._idx_q_goal).CopyToVector()

        # Determine if we should solve IK again
        need_solve = True
        if np.isfinite(last_target).all():
            if np.linalg.norm(target - last_target) <= self._target_tol:
                need_solve = False

        if not need_solve:
            # Keep q_goal as-is, keep meta as-is.
            return

        # Seed choice: measured, or blended measured+previous
        if np.isfinite(q_goal_prev).all():
            q_seed = (1.0 - self._seed_blend) * q_meas + self._seed_blend * q_goal_prev
        else:
            q_seed = q_meas

        t0 = time.time()
        q_sol, ok = self._solve_ik(target_xyz=target, q_seed=q_seed)
        t1 = time.time()

        # Update state
        discrete_state.get_mutable_vector(self._idx_last_target).set_value(target)

        if ok:
            discrete_state.get_mutable_vector(self._idx_q_goal).set_value(q_sol)
            discrete_state.get_mutable_vector(self._idx_meta).set_value(
                np.array([1.0, float(t1 - t0)], dtype=float)
            )
        else:
            # If IK fails, keep the previous q_goal (don’t jerk the robot)
            discrete_state.get_mutable_vector(self._idx_meta).set_value(
                np.array([0.0, float(t1 - t0)], dtype=float)
            )

    def _solve_ik(self, target_xyz, q_seed):
        """
        Cached AutoDiff version of your working IK:
        - position box (tol_xy/tol_z)
        - tool +z aligned with world -z within ang_tol
        - soft position-centering cost via cached AutoDiff plant/context
        """
        plant = self._plant
        model_instance = self._model
        tool_frame = self._tool

        q_seed = np.asarray(q_seed, dtype=float).reshape(-1)
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)

        # Seed context (Float plant) at q_seed
        plant_context = plant.CreateDefaultContext()
        plant.SetPositions(plant_context, model_instance, q_seed)

        ik = InverseKinematics(plant, plant_context)
        q = ik.q()
        prog = ik.prog()

        # --- tolerances: identical to your working version ---
        tol_xy = 4e-3
        tol_z  = 1.5e-3

        lower = target_xyz + np.array([-tol_xy, -tol_xy, -tol_z])
        upper = target_xyz + np.array([+tol_xy, +tol_xy, +tol_z])

        ik.AddPositionConstraint(
            frameB=tool_frame,
            p_BQ=np.zeros(3),
            frameA=plant.world_frame(),
            p_AQ_lower=lower,
            p_AQ_upper=upper,
        )

        # Axis alignment: tool +z aligned with world -z (yaw free)
        theta = float(self._ang_tol_deg) * np.pi / 180.0
        na_A = np.array([[0.0], [0.0], [-1.0]])  # world -z
        nb_B = np.array([[0.0], [0.0], [ 1.0]])  # tool +z
        ik.AddAngleBetweenVectorsConstraint(
            plant.world_frame(), na_A,
            tool_frame, nb_B,
            0.0, theta
        )

        # Soft cost: prefer center of the box (minimize ||p(q)-target||^2)
        w_pos = 1000.0

        plant_ad = self._plant_ad
        ctx_ad = self._ctx_ad
        model_ad = self._model_ad
        tool_ad = self._tool_ad

        def pos_cost(x):
            # x can be float ndarray or AutoDiffXd ndarray (SNOPT).
            if getattr(x, "dtype", None) == object:
                q_ad = x
            else:
                x = np.asarray(x, dtype=float).reshape((-1, 1))
                q_ad = InitializeAutoDiff(x, num_derivatives=x.shape[0]).reshape((-1,))

            plant_ad.SetPositions(ctx_ad, model_ad, q_ad)
            X_WT = plant_ad.CalcRelativeTransform(ctx_ad, plant_ad.world_frame(), tool_ad)
            p = X_WT.translation()
            e = p - target_xyz
            return w_pos * e.dot(e)

        prog.AddCost(pos_cost, q)

        # Seed
        prog.SetInitialGuess(q, q_seed)

        result = Solve(prog)
        if not result.is_success():
            return q_seed, False

        return np.asarray(result.GetSolution(q), dtype=float), True


def main_smoke():
    station = MakeIiwaRealStation(
        lcm_channel_suffix="",  # set if needed
        tool_offset_z=0.10,
        max_delta_per_step=0.01,
    )
    ctx = station.CreateDefaultContext()
    sim = Simulator(station, ctx)
    sim.set_target_realtime_rate(1.0)

    # Example: hold a nominal posture
    q_nom = np.array([0, np.pi/2, -np.pi/2, 0, 0, 0, 0], dtype=float)
    station.GetInputPort("q_goal").FixValue(ctx, q_nom)

    while ctx.get_time() < 5.0:
        sim.AdvanceTo(ctx.get_time() + 0.1)

        tool = station.GetOutputPort("tool_metrics").Eval(ctx)
        # tool = [x, y, z, align]
        print(f"t={ctx.get_time():.2f} tool xyz=({tool[0]:+.3f},{tool[1]:+.3f},{tool[2]:+.3f}) align={tool[3]:+.3f}")

def run_station_fixed_target(
    station,
    target_xyz,
    duration_sec=10.0,
    print_dt=0.2,
    check_dt=0.05,
    hold_count_required=5,
    xy_tol=5e-3,
    z_tol=3e-3,
    align_tol=0.985,
):
    ctx = station.CreateDefaultContext()
    sim = Simulator(station, ctx)
    sim.set_target_realtime_rate(1.0)

    target_xyz = np.asarray(target_xyz, float).reshape(3)
    station.GetInputPort("target_xyz").FixValue(ctx, target_xyz)

    t_next_print = 0.0
    t_next_check = 0.0
    hold = 0

    while ctx.get_time() < duration_sec:
        t = ctx.get_time()
        sim.AdvanceTo(min(duration_sec, t + check_dt))

        tool = station.GetOutputPort("tool_metrics").Eval(ctx)   # [x,y,z,align]
        ik_status = station.GetOutputPort("ik_status").Eval(ctx) # [success, solve_time]

        # Check "reached" at check_dt cadence
        if ctx.get_time() >= t_next_check:
            if is_reached(tool, target_xyz, xy_tol=xy_tol, z_tol=z_tol, align_tol=align_tol):
                hold += 1
            else:
                hold = 0
            t_next_check += check_dt

        # Print at print_dt cadence
        if ctx.get_time() >= t_next_print:
            p = tool[:3]
            align = float(tool[3])
            xy_err = float(np.linalg.norm(p[:2] - target_xyz[:2]))
            z_err = float(abs(p[2] - target_xyz[2]))
            print(
                f"t={ctx.get_time():.2f} "
                f"tool=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) "
                f"xy_err={xy_err*1e3:.1f}mm z_err={z_err*1e3:.1f}mm align={align:+.3f} "
                f"IK_ok={int(ik_status[0])} IK_dt={ik_status[1]*1e3:.1f}ms "
                f"hold={hold}/{hold_count_required}"
            )
            t_next_print += print_dt

        # Stop early if held long enough
        if hold >= hold_count_required:
            print(f"[DONE] Reached target for {hold_count_required} consecutive checks.")
            break

    return {
        "t_end": float(ctx.get_time()),
        "reached": bool(hold >= hold_count_required),
        "hold": int(hold),
    }

def is_reached(tool_metrics, target_xyz, xy_tol=5e-3, z_tol=3e-3, align_tol=0.985):
    p = tool_metrics[:3]
    align = float(tool_metrics[3])
    target_xyz = np.asarray(target_xyz, float)

    xy_err = np.linalg.norm(p[:2] - target_xyz[:2])
    z_err = abs(p[2] - target_xyz[2])
    return (xy_err <= xy_tol) and (z_err <= z_tol) and (align >= align_tol)


if __name__ == "__main__":
    main_smoke()
