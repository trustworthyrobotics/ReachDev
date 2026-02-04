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

if __name__ == "__main__":
    main_smoke()
