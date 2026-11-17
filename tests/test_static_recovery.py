"""Local behavioral checks; no simulator, robot motion or model request."""

import base64
import importlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from static_recovery import StaticRecovery


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aV1cAAAAASUVORK5CYII="
)


@dataclass
class Arm:
    tcp_pos: tuple = (0.1, -0.1, 0.9)
    pitch: float = 60.0
    yaw: float = 0.0
    roll: float = 0.0
    gripper: float = 0.4
    joints: tuple = (0.0,) * 6


@dataclass
class State:
    step: int = 7
    left: Arm = field(default_factory=Arm)
    right: Arm = field(default_factory=Arm)


class Ledger(list):
    @property
    def records(self):
        return self


class ReadOnlyRobot:
    ABSENT_MEMBERS = ("done",)

    def __init__(self):
        self.calls = []
        self.missing_camera = None
        self._ledger = Ledger()
        self._backend = SimpleNamespace(env_steps=lambda: 0)

    def state(self, step=-1):
        self.calls.append(("state",) if step == -1 else ("state", step))
        if step > 7:
            raise FileNotFoundError("historical step unavailable")
        return State(step=7 if step == -1 else step)

    def artifact(self, name, *, step):
        self.calls.append(("artifact", name, step))
        if name == self.missing_camera:
            raise RuntimeError("camera unavailable")
        return PNG

    def camera_meta(self, camera, *, step):
        self.calls.append(("camera_meta", camera, step))
        return {"camera": camera, "step": step, "has_depth": False}

    def _take_show_requests(self):
        return []

    def _show_artifacts(self, requested):
        return []

    def _reported_success(self):
        return False

    def move_to(self, *args, **kwargs):
        raise AssertionError("the diagnosis program must not move the robot")


@pytest.fixture
def case(tmp_path):
    shown = []
    robot = ReadOnlyRobot()
    note = StaticRecovery(
        robot,
        open_file=lambda name, *args, **kwargs: open(tmp_path / name, *args, **kwargs),
        display_image=shown.append,
    )
    return note, robot, shown


def begin(note):
    return note.begin("lift", "object stayed on table", recovery_goal="object held and lifted, confirmed by tool and image",
                      evidence=["cell 5: head image"])


def classify(note):
    return note.classify(
        "missed grasp", "position may be stale",
        evidence=["cell 5: object stayed on table"],
        question="Where is the object relative to the fingers now?",
    )


def conclude(note):
    return note.conclude(correction='Use the updated measured center for the next approach.', next_segment='Approach above the measured center; stop before descent.', continue_when='Target is visibly centered and the descent path is clear.')


def test_record_preserves_initial_diagnosis_and_revisions(case, tmp_path):
    note, robot, shown = case
    assert begin(note)["phase"] == "reported"
    assert classify(note)["phase"] == "classified"
    measured = {"locate": {"views": 1, "note": "occluded"}}
    result = note.observe(cameras=("cam_head", "cam_left_wrist"), measurements=measured)
    measured["locate"]["views"] = 3
    assert result["measurements"]["locate"]["views"] == 1
    assert result["recorded_step"] == 7
    assert "joints" not in result["state"]["left"]
    assert shown == [PNG, PNG]
    assert all(call[-1] == 7 for call in robot.calls if call[0] != "state")
    for path in result["images"]:
        assert (tmp_path / path).read_bytes() == PNG
    conclude(note)
    note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="failed", evidence=["cell 8: original grasp check failed"])
    note.classify(
        "unknown", "position alone did not explain the failure",
        evidence=["cell 8: original check"], question="Is the visible grasp surface suitable?",
    )
    text = (tmp_path / note.record).read_text(encoding="utf-8")
    assert text.count("## Provisional diagnosis") == 2
    assert "position may be stale" in text and "position alone did not explain" in text
    assert "Retry outcome" in text
    assert "agent_supplied_measurements" in text
    assert "camera_meta" in {call[0] for call in robot.calls}


def test_existing_evidence_can_avoid_new_camera_reads(case, tmp_path):
    note, robot, shown = case
    begin(note)
    classify(note)
    note.conclude(correction='Inspect the visible grasp surface before selecting a retry.', next_segment='Return the existing grasp view.', continue_when='The grasp surface is visible.')
    assert robot.calls == [("state",), ("state",)] and shown == []
    assert note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="uncertain", evidence=["original check inconclusive"])["phase"] == "retry_failed"


def test_unknown_cause_trial_snapshots_arguments_and_uses_observation_index(case, tmp_path):
    note, _, _ = case
    begin(note)
    note.classify("unknown", "target may be stale", evidence=["object stayed on table"],
                  question="Does remeasurement change the target?")
    note.conclude("Remeasure to test stale targeting.", next_segment="Read target and inspect view.",
                  continue_when="Locate and view agree on the target.")
    args = {"arm": "left", "pos": [0.1, -0.1, 0.9]}
    receipt = {"reached": False, "blocked": True}
    obs = note.observe(measurements={"arguments": args, "move": receipt})
    args["pos"][0] = 9
    receipt["reached"] = True
    assert obs["measurements"]["arguments"]["pos"][0] == 0.1
    assert obs["measurements"]["move"]["reached"] is False
    note.outcome("uncertain", evidence=["current view does not settle targeting"],
                 evidence_step=obs["recorded_step"], assessment="Cause remains unknown.")
    entries = [json.loads(v) for v in re.findall(r'```json\n(.*?)\n```',
               (tmp_path / note.record).read_text(encoding="utf-8"), re.S)]
    observation = next(v for v in entries if "agent_supplied_measurements" in v)
    assert observation["agent_supplied_measurements"] == obs["measurements"]
    assert entries[-1]["evidence_step"] == observation["recorded_step"]


def test_localize_finds_first_unresolved_and_preserves_revision(case, tmp_path):
    note, robot, shown = case
    begin(note)
    steps = [
        {"action": "lift", "status": "passed", "evidence": ["cell 1: object lifted"]},
        {"action": "rotate while holding", "status": "uncertain", "evidence": ["cell 2: grip occluded"]},
        {"action": "arm arrived", "status": "passed", "evidence": ["cell 2: reached"]},
        {"action": "insert", "status": "not_attempted", "evidence": []},
    ]
    result = note.localize(steps=steps)
    assert result["phase"] == "localized"
    assert result["scope"]["first_unresolved"]["step"] == 2
    assert result["scope"]["last_confirmed"]["step"] == 1
    assert result["scope"]["not_attempted"] == [4]
    steps[1]["evidence"].append("later mutation")
    assert "later mutation" not in str(result)
    classify(note)
    note.observe(cameras=("cam_head",))
    conclude(note)
    note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="failed", evidence=["cell 4: object dropped"])
    changed = note.localize(steps=[{"action": "retain grasp", "status": "failed", "evidence": ["cell 4: dropped"]}])
    assert changed["scope"]["last_confirmed"] is None
    assert changed["scope"]["first_unresolved"]["step"] == 1
    assert "question" not in changed and "error_type" not in changed
    assert note._question is None and note._error_type is None
    text = (tmp_path / note.record).read_text(encoding="utf-8")
    assert text.count("## Local action scope") == 2
    assert "grip occluded" in text and "cell 4: dropped" in text
    assert shown == [PNG]
    assert all(call[0] in ("state", "artifact", "camera_meta") for call in robot.calls)


@pytest.mark.parametrize("steps", [
    [], "lift", ["lift"],
    [{"action": "lift", "status": "success", "evidence": ["image"]}],
    [{"action": "lift", "status": "passed", "evidence": []}],
    [{"action": "", "status": "not_attempted", "evidence": []}],
])
def test_invalid_local_scope_does_not_change_record(case, tmp_path, steps):
    note, robot, _ = case
    begin(note)
    before = (tmp_path / note.record).read_bytes()
    with pytest.raises(ValueError):
        note.localize(steps=steps)
    assert (tmp_path / note.record).read_bytes() == before
    assert note._phase == "reported" and robot.calls == [("state",)]


def test_localize_does_not_invent_failure_or_claim_current_success(case):
    note, _, _ = case
    begin(note)
    result = note.localize(steps=[{"action": "insert", "status": "not_attempted", "evidence": []}])
    assert result["scope"]["first_unresolved"] is None
    assert result["scope"]["last_confirmed"] is None
    assert result["phase"] == "localized"


def test_unique_files_never_overwrite_a_previous_error(case, tmp_path):
    note, _, _ = case
    (tmp_path / "failure_001.md").write_text("previous record", encoding="utf-8")
    assert begin(note)["record"] == "failure_002.md"
    assert (tmp_path / "failure_001.md").read_text() == "previous record"


@pytest.mark.parametrize("cameras", ["cam_head", ("../outside",), ("cam_head", "cam_head")])
def test_invalid_cameras_do_not_collect_or_write(case, tmp_path, cameras):
    note, robot, _ = case
    begin(note)
    classify(note)
    before = (tmp_path / note.record).read_bytes()
    with pytest.raises(ValueError):
        note.observe(cameras=cameras)
    assert robot.calls == [("state",)]
    assert (tmp_path / note.record).read_bytes() == before


def test_camera_error_is_recorded_and_not_replaced_by_a_success(case, tmp_path):
    note, robot, shown = case
    begin(note)
    classify(note)
    robot.missing_camera = "cam_left_wrist.png"
    with pytest.raises(RuntimeError, match="camera unavailable"):
        note.observe(cameras=("cam_head", "cam_left_wrist"))
    assert shown == []
    text = (tmp_path / note.record).read_text(encoding="utf-8")
    assert "Observation error" in text and "## Static observation" not in text
    robot.missing_camera = None
    recovered = note.observe(cameras=("cam_head",))
    assert "observation_002" in recovered["images"][0]


def test_invalid_phase_and_empty_evidence_are_not_silently_accepted(case, tmp_path):
    note, _, _ = case
    with pytest.raises(ValueError):
        classify(note)
    with pytest.raises(ValueError):
        note.begin("lift", "failed", recovery_goal="object held", evidence=[])
    assert not list(tmp_path.iterdir())
    begin(note)
    with pytest.raises(ValueError):
        begin(note)
    with pytest.raises(ValueError):
        note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="passed", evidence=["not checked yet"])


def test_program_runs_in_original_cells_and_delivers_actual_images(tmp_path, monkeypatch):
    # The author host is supplied by the pinned checkout documented in README.
    root = Path(__file__).resolve().parents[1]
    from pyrualean.arms import ArmToolkit

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copyfile(root / "src/static_recovery.py", workspace / "static_recovery.py")
    monkeypatch.syspath_prepend(str(workspace))
    monkeypatch.delitem(sys.modules, "static_recovery", raising=False)
    importlib.invalidate_caches()
    robot = ReadOnlyRobot()
    host = ArmToolkit(robot, arm="cells", workspace=workspace, images="on-demand", feedback="pure")
    first = host.execute_tool("python", {"code": (
        "from static_recovery import StaticRecovery\n"
        "case = StaticRecovery(robo, open_file=open, display_image=display)\n"
        "reply = case.begin('lift', 'object stayed on table', recovery_goal='object held and lifted', evidence=['cell 0: head view'])\n"
        "continued = True"
    )})
    assert "exception" not in first.result
    assert "localize(steps=" in first.result["stdout"]
    localized = host.execute_tool("python", {"code": (
        "print(case.localize(steps=[{'action':'lift', 'status':'failed', "
        "'evidence':['cell 0: object on table']}]))"
    )})
    assert "exception" not in localized.result
    assert "first_unresolved" in localized.result["stdout"]
    reviewed = host.execute_tool("python", {"code": (
        "print(case.review(question='When was the hold lost?', condition='object held', "
        "checkpoints=[{'step': 3, 'action': 'after lift', 'cameras': ('cam_left_wrist',)}, "
        "{'step': 5, 'action': 'after approach', 'cameras': ('cam_head', 'cam_left_wrist')}]))"
    )})
    assert "exception" not in reviewed.result
    assert reviewed.images == [PNG, PNG, PNG]
    assert "last established and first violated" in reviewed.result["stdout"]
    assert (workspace / "failure_001_review_001_step_0003_cam_left_wrist.png").read_bytes() == PNG
    second = host.execute_tool("python", {"code": (
        "print(case.classify('unknown', 'target may be stale', evidence=['cell 0: image'], "
        "question='Where is the target now?'))"
    )})
    assert "exception" not in second.result
    third = host.execute_tool("python", {"code": (
        "print(case.observe(cameras=('cam_head', 'cam_left_wrist')))"
    )})
    assert "exception" not in third.result
    assert third.images == [PNG, PNG]
    assert (workspace / "failure_001.md").exists()
    assert not host.sandbox_flags
    planned = host.execute_tool("python", {"code": (
        "print(case.conclude(correction='Check current hold', next_segment='Read state only', continue_when='Target visibly held'))"
    )})
    assert "exception" not in planned.result
    judged = host.execute_tool("python", {"code": (
        "print(case.outcome('uncertain', evidence=['step 7 head: hold unclear'], evidence_step=7, "
        "assessment='View does not establish target held; no contrary evidence resolves this.'))"
    )})
    assert "exception" not in judged.result and "retry_failed" in judged.result["stdout"]
    combined = host.execute_tool("python", {"code": (
        'batch = StaticRecovery(robo, open_file=open, display_image=display)\n'
        "batch.begin('inspect grip', 'current hold unverified', recovery_goal='current hold checked', evidence=['prior view unclear'])\n"
        "scope = batch.localize(steps=[{'action':'check hold', 'status':'uncertain', 'evidence':['prior view unclear']}])\n"
        "if scope['scope']['first_unresolved'] is not None:\n"
        "    batch.classify('unknown', 'hold not established', evidence=['prior view unclear'], question='What is the current hold?')\n"
        "    batch.conclude(correction='Read current opening, then inspect view', next_segment='Read state', continue_when='Opening reading is available')\n"
        '    current = robo.state()\n'
        '    if 0 <= current.left.gripper <= 1:\n'
        "        batch.outcome('progress', evidence=['current opening recorded'], evidence_step=current.step, assessment='Opening available; hold still unverified. No contrary reading.')\n"
        "        batch.conclude(correction='Inspect wrist', next_segment='Return wrist image', continue_when='Target visibly held')\n"
        "        print(batch.observe(cameras=('cam_left_wrist',)))"
    )})
    assert "exception" not in combined.result and combined.images == [PNG]
    assert "'phase': 'concluded'" in combined.result["stdout"]
    record = (workspace / "failure_002.md").read_text(encoding="utf-8")
    assert record.count("## Revised diagnosis and proposed correction") == 2
    assert record.count("## Retry outcome reported by agent") == 1
    # No model is invoked here; this exercises continuation after the image-return boundary.
    continued = host.execute_tool("python", {"code": (
        "batch.outcome('uncertain', evidence=['returned view: hold unresolved'], evidence_step=robo.state().step, "
        "assessment='Test supplies an inconclusive visual assessment; no confirmation.')\n"
        "batch.classify('unknown', 'view insufficient', evidence=['returned view'], question='Can the head view clarify?')\n"
        "print(batch.observe(cameras=('cam_head',)))"
    )})
    assert "exception" not in continued.result and continued.images == [PNG]
    assert "'phase': 'observed'" in continued.result["stdout"]
    assert not host.sandbox_flags
    rejected = host.execute_tool("python", {"code": (
        "batch.conclude('Try revised yaw', next_segment='Retry grasp', continue_when='Object held')\n"
        "batch.conclude('Use current center', next_segment='Retry grasp', continue_when='Object held')\n"
        "reply = batch.outcome('passed', evidence=['old grip'], evidence_step=6)\n"
        "obs = batch.observe(cameras=(), measurements={'target': np.array([0.1, 0.2, 0.8])})\n"
        "print('cell continued', reply['phase'], obs['images'])"
    )})
    assert "exception" not in rejected.result and not rejected.images
    assert "Outcome not recorded" in rejected.result["stdout"]
    assert "cell continued concluded []" in rejected.result["stdout"]
    assert "'target':" not in rejected.result["stdout"]
    assert (workspace / "failure_002.md").read_text().count("## Retry outcome reported by agent") == 2
    crash = host.execute_tool("python", {"code": "raise RuntimeError('original check')"})
    assert "RuntimeError: original check" in crash.result["exception"]
    assert not host.finish


def test_review_uses_sorted_historical_state_and_images(case, tmp_path):
    note, robot, shown = case
    begin(note)
    note.localize(steps=[{"action": "handover", "status": "uncertain", "evidence": ["cell 5"]}])
    robot.artifact = lambda name, *, step: PNG + str(step).encode()
    result = note.review(question="When was the hold lost?", condition="key held", checkpoints=[
        {"step": 5, "action": "receiver approach", "cameras": ["cam_head", "cam_left_wrist"]},
        {"step": 3, "action": "donor lift", "cameras": ["cam_left_wrist"]},
    ])
    assert [p["step"] for p in result["checkpoints"]] == [3, 5]
    assert shown == [PNG + b"3", PNG + b"5", PNG + b"5"]
    assert result["phase"] == "localized"
    assert result["question"] == "When was the hold lost?"
    assert ("state", 3) in robot.calls and ("state", 5) in robot.calls
    text = (tmp_path / note.record).read_text(encoding="utf-8")
    assert '"step": 3' in text and '"step": 5' in text and '"calibration"' in text
    for point in result["checkpoints"]:
        assert "joints" not in point["state"]["left"]
        for path in point["images"]:
            assert (tmp_path / path).read_bytes() == PNG + str(point["step"]).encode()
    classify(note)
    conclude(note)


@pytest.mark.parametrize("checkpoints", [
    [], [{"step": -1, "action": "current", "cameras": ["cam_head"]}],
    [{"step": True, "action": "lift", "cameras": ["cam_head"]}],
    [{"step": 3, "action": "lift", "cameras": ["unknown"]}],
    [{"step": 3, "action": "lift", "cameras": ["cam_head"]}] * 2,
])
def test_review_rejects_ambiguous_indices_before_reads(case, checkpoints):
    note, robot, shown = case
    begin(note)
    classify(note)
    with pytest.raises(ValueError):
        note.review(question="When?", condition="held", checkpoints=checkpoints)
    assert robot.calls == [("state",)] and not shown


def test_review_missing_history_is_not_replaced_by_current_view(case, tmp_path):
    note, robot, shown = case
    begin(note)
    classify(note)
    with pytest.raises(FileNotFoundError):
        note.review(question="When?", condition="held", checkpoints=[
            {"step": 8, "action": "missing", "cameras": ["cam_head"]},
        ])
    assert not shown
    assert robot.calls == [("state",), ("state", 8)]
    assert "Boundary review error" in (tmp_path / note.record).read_text(encoding="utf-8")
    result = note.review(question="When?", condition="held", checkpoints=[
        {"step": 0, "action": "initial", "cameras": ["cam_head"]},
    ])
    assert "review_002" in result["checkpoints"][0]["images"][0]


def test_review_rejects_backend_returning_a_different_state(case, tmp_path):
    note, robot, shown = case
    begin(note)
    classify(note)
    robot.state = lambda step=-1: State(step=7)
    with pytest.raises(ValueError, match="does not match"):
        note.review(question="When?", condition="held", checkpoints=[
            {"step": 3, "action": "lift", "cameras": ["cam_head"]},
        ])
    assert not shown
    assert not list(tmp_path.glob("*.png"))


def test_review_does_not_deliver_incomplete_comparison(case, tmp_path):
    note, robot, shown = case
    begin(note)
    classify(note)
    robot.missing_camera = "cam_left_wrist.png"
    with pytest.raises(RuntimeError, match="camera unavailable"):
        note.review(question="When?", condition="held", checkpoints=[
            {"step": 3, "action": "lift", "cameras": ["cam_head"]},
            {"step": 5, "action": "approach", "cameras": ["cam_left_wrist"]},
        ])
    assert not shown
    text = (tmp_path / note.record).read_text(encoding="utf-8")
    assert "Boundary review error" in text and "## Boundary review 1" not in text


def test_static_observation_can_lead_to_historical_review(case):
    note, robot, shown = case
    begin(note)
    classify(note)
    note.observe(cameras=["cam_head"])
    shown.clear()
    reply = note.review(question="Did the target move?", condition="target at prior position", checkpoints=[
        {"step": 3, "action": "before retry", "cameras": ["cam_head"]},
        {"step": 7, "action": "after retry", "cameras": ["cam_head"]},
    ])
    assert reply["phase"] == "observed" and shown == [PNG, PNG]
    note.classify("unknown", "target may have moved", evidence=["review steps 3 and 7"],
                  question="Where is the target now?")
    conclude(note)


def test_intermediate_progress_keeps_fixed_goal_open_until_restored(case, tmp_path):
    note, robot, shown = case
    goal = begin(note)["recovery_goal"]
    classify(note)
    plan = conclude(note)
    assert plan["planned_segment"]["next_segment"].endswith("stop before descent.")
    progress = note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="progress", evidence=["cell 6: view cleared; target visible but not held"])
    assert progress["phase"] == "progress" and progress["recovery_goal"] == goal
    assert "planned_segment" not in progress
    note.observe(cameras=["cam_left_wrist"])
    second = note.conclude(correction='Approach contact before closing.', next_segment='Descend, then return.', continue_when='Object is between open pads.')
    assert second["recovery_goal"] == goal
    assert note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="progress", evidence=["cell 8: contact aligned, not yet closed"])["phase"] == "progress"
    third = note.conclude(correction='Close, check then lift.', next_segment='Close and lift only if holding; return images.', continue_when='The fixed recovery goal is established.')
    assert third["recovery_goal"] == goal
    final = note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="passed", evidence=["cell 9: holding true and object visibly lifted"])
    assert final["phase"] == "checked" and final["recovery_goal"] == goal
    with pytest.raises(ValueError):
        note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="passed", evidence=["duplicate"])
    entries = [json.loads(s) for s in re.findall(r'```json\n(.*?)\n```',
               (tmp_path / note.record).read_text(encoding="utf-8"), re.S)]
    results = [d for d in entries if "result" in d]
    assert [d["result"] for d in results] == ["progress", "progress", "passed"]
    assert all(d["recovery_goal"] == goal and d["planned_segment"] for d in results)
    assert shown == [PNG] and all(c[0] in ("state", "artifact", "camera_meta") for c in robot.calls)


@pytest.mark.parametrize("field", ["correction", "next_segment", "continue_when"])
def test_incomplete_segment_does_not_commit_a_plan(case, tmp_path, field):
    note, _, _ = case
    begin(note)
    classify(note)
    before = (tmp_path / note.record).read_bytes()
    args = dict(correction="reobserve to test visibility",
                next_segment="show", continue_when="target visible in image")
    args[field] = " "
    with pytest.raises(ValueError, match=field):
        note.conclude(**args)
    assert note._phase == "classified" and note._segment is None
    assert (tmp_path / note.record).read_bytes() == before


def test_empty_recovery_goal_is_rejected_before_registration(case, tmp_path):
    note, robot, _ = case
    with pytest.raises(ValueError, match="recovery_goal"):
        note.begin("pickup", "empty grasp", recovery_goal=" ", evidence=["cell 3"])
    assert not list(tmp_path.iterdir()) and not robot.calls


def test_progress_can_reenter_diagnosis_without_changing_goal(case):
    note, _, _ = case
    goal = begin(note)["recovery_goal"]
    classify(note)
    conclude(note)
    note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="progress", evidence=["view cleared"])
    revised = note.localize(steps=[{"action": "retain prior support", "status": "uncertain", "evidence": ["new image"]}])
    assert "recovery_goal" not in revised
    assert "recovery_goal" not in classify(note)
    assert conclude(note)["recovery_goal"] == goal


def test_failed_retry_can_review_before_revising_diagnosis(case):
    note, _, shown = case
    goal = begin(note)["recovery_goal"]
    classify(note)
    conclude(note)
    note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="failed", evidence=["cell 16: empty after lift"])
    reply = note.review(question="When was the hold lost?", condition="key held", checkpoints=[
        {"step": 3, "action": "close", "cameras": ["cam_left_wrist"]},
        {"step": 7, "action": "lift", "cameras": ["cam_left_wrist"]},
    ])
    assert reply["phase"] == "retry_failed" and "recovery_goal" not in reply
    assert shown == [PNG, PNG]
    classify(note)
    assert conclude(note)["recovery_goal"] == goal


def test_retry_observation_preserves_segment_for_outcome(case, tmp_path):
    note, robot, shown = case
    goal = begin(note)["recovery_goal"]
    classify(note)
    plan = conclude(note)["planned_segment"]
    robot.missing_camera = "cam_head.png"
    with pytest.raises(RuntimeError, match="camera unavailable"):
        note.observe(cameras=["cam_head"])
    assert note._phase == "concluded" and note._segment == plan
    robot.missing_camera = None
    for camera in ("cam_head", "cam_left_wrist"):
        reply = note.observe(cameras=[camera])
        assert reply["phase"] == "concluded" and reply["planned_segment"] == plan
        assert reply["recovery_goal"] == goal
        assert "continue_when" in reply["prompt"] and "outcome" in reply["prompt"]
    assert shown == [PNG, PNG]
    assert note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="progress", evidence=["cell 20: target centered, grasp pending"])["phase"] == "progress"
    records = re.findall(r'```json\n(.*?)\n```', (tmp_path / note.record).read_text(encoding="utf-8"), re.S)
    assert json.loads(records[-1])["planned_segment"] == plan


def test_closed_recovery_still_rejects_observe_and_review(case):
    note, robot, _ = case
    begin(note)
    classify(note)
    conclude(note)
    note.outcome(evidence_step=7, assessment="Current referenced evidence supports this result; no contrary evidence observed.", result="passed", evidence=["cell 9: object visibly held and lifted"])
    before = list(robot.calls)
    with pytest.raises(ValueError):
        note.observe(cameras=["cam_head"])
    with pytest.raises(ValueError):
        note.review(question="When?", condition="held", checkpoints=[
            {"step": 7, "action": "lift", "cameras": ["cam_head"]},
        ])
    assert robot.calls == before


def test_begin_after_passed_preserves_old_files_and_resets_case(case, tmp_path):
    note, _, _ = case
    begin(note)
    note.localize(steps=[{"action": "lift", "status": "failed", "evidence": ["old view"]}])
    classify(note)
    note.review(question="When?", condition="held", checkpoints=[
        {"step": 3, "action": "lift", "cameras": ["cam_head"]}])
    note.observe()
    conclude(note)
    note.outcome("passed", evidence=["current view"], evidence_step=7, assessment="Goal restored.")
    old_files = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    reply = note.begin("place", "target missed", recovery_goal="target placed", evidence=["new view"])
    assert reply["record"] == "failure_002.md" and reply["phase"] == "reported"
    assert note._observation == note._review == 0
    assert note._question is note._error_type is note._scope is note._segment is None
    assert reply["recovery_goal"] == "target placed"
    classify(note)
    obs = note.observe()
    assert obs["images"] == ["failure_002_observation_001_cam_head.png"]
    assert all((tmp_path / name).read_bytes() == data for name, data in old_files.items())


@pytest.mark.parametrize("result", ["progress", "failed", "uncertain"])
def test_begin_cannot_replace_unresolved_case(case, tmp_path, result):
    note, robot, _ = case
    begin(note)
    classify(note)
    conclude(note)
    note.outcome(result, evidence=["current view"], evidence_step=7, assessment="Goal unresolved.")
    before = dict(vars(note)), list(robot.calls), {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        begin(note)
    assert (dict(vars(note)), robot.calls, {p.name: p.read_bytes() for p in tmp_path.iterdir()}) == before


def test_failed_new_registration_preserves_closed_case(case, tmp_path):
    note, _, _ = case
    begin(note)
    classify(note)
    conclude(note)
    note.outcome("passed", evidence=["current view"], evidence_step=7, assessment="Goal restored.")
    before = dict(vars(note)), {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="recovery_goal"):
        note.begin("place", "missed", recovery_goal="", evidence=["new view"])
    assert (dict(vars(note)), {p.name: p.read_bytes() for p in tmp_path.iterdir()}) == before

    def failing_open(*args, **kwargs):
        raise OSError("record unavailable")

    note.open_file = failing_open
    before_state = dict(vars(note))
    with pytest.raises(OSError, match="record unavailable"):
        begin(note)
    assert vars(note) == before_state
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before[1]


@pytest.mark.parametrize("step", [8, True, -1])
def test_outcome_rejects_future_or_invalid_evidence_without_closing(case, tmp_path, step):
    note, _, _ = case
    begin(note)
    classify(note)
    plan = conclude(note)["planned_segment"]
    before = (tmp_path / note.record).read_bytes()
    with pytest.raises(ValueError):
        note.outcome("passed", evidence=["earlier holding=True"], evidence_step=step,
                     assessment="Claimed hold based on earlier close.")
    assert note._phase == "concluded" and note._segment == plan
    assert (tmp_path / note.record).read_bytes() == before


def test_current_contradictory_evidence_can_record_uncertain_without_new_capture(case, tmp_path):
    note, robot, shown = case
    goal = begin(note)["recovery_goal"]
    classify(note)
    plan = conclude(note)["planned_segment"]
    robot.state = lambda step=-1: State(step=8, left=Arm(gripper=0.0))
    reply = note.outcome("uncertain", evidence=["step 8: opening zero, target obscured; earlier close holding=True"],
        evidence_step=8, assessment="Current evidence does not establish retained hold; prior close conflicts with zero opening and the view is obscured.")
    assert reply["phase"] == "retry_failed" and reply["recovery_goal"] == goal
    assert not shown
    entries = re.findall(r'```json\n(.*?)\n```', (tmp_path / note.record).read_text(encoding="utf-8"), re.S)
    last = json.loads(entries[-1])
    assert last["evidence_step"] == last["state_at_judgment"]["step"] == 8
    assert last["planned_segment"] == plan and plan["start_step"] == 7
    assert last["state_at_judgment"]["left"]["gripper"] == 0.0
    assert "conflicts" in last["assessment"]


def test_outcome_rejects_explicit_empty_assessment_before_recording(case, tmp_path):
    note, _, _ = case
    begin(note)
    classify(note)
    conclude(note)
    before = (tmp_path / note.record).read_bytes()
    with pytest.raises(ValueError, match="assessment"):
        note.outcome("passed", evidence=["current view"], evidence_step=7, assessment=" ")
    assert note._phase == "concluded" and (tmp_path / note.record).read_bytes() == before


def test_numeric_evidence_is_snapshotted_without_manual_conversion(case):
    note, _, _ = case
    begin(note)
    classify(note)
    xyz = np.array([0.1, -0.2, 0.8])
    reply = note.observe(measurements={"target": xyz, "count": np.int64(2),
                                      "valid": np.bool_(True), "pose": Arm(tcp_pos=xyz)})
    xyz[0] = 9
    assert reply["measurements"] == {"target": [0.1, -0.2, 0.8], "count": 2,
                                     "valid": True, "pose": {**vars(Arm()), "tcp_pos": [0.1, -0.2, 0.8], "joints": [0.0] * 6}}


@pytest.mark.parametrize("value", [np.array([float('nan')]), np.float32(float('inf')), object()])
def test_invalid_numeric_evidence_does_not_write_or_capture(case, tmp_path, value):
    note, robot, shown = case
    begin(note)
    classify(note)
    before = (tmp_path / note.record).read_bytes()
    calls = list(robot.calls)
    with pytest.raises((ValueError, TypeError)):
        note.observe(measurements={"invalid": value})
    assert (tmp_path / note.record).read_bytes() == before and robot.calls == calls and not shown


def test_measurements_only_observation_does_not_invent_images(case):
    note, robot, shown = case
    begin(note)
    classify(note)
    reply = note.observe(cameras=(), measurements={"holding": False})
    assert reply["images"] == [] and reply["recorded_step"] == 7
    assert reply["measurements"] == {"holding": False} and not shown
    assert all(call[0] == "state" for call in robot.calls)
    with pytest.raises(ValueError):
        note.review(question="When?", condition="held", checkpoints=[
            {"step": 7, "action": "lift", "cameras": ()}])


def test_optional_explanations_are_not_fabricated(case, tmp_path):
    note, _, _ = case
    begin(note)
    note.classify("unknown", "contact uncertain", evidence=["current view"])
    assert note._question is None
    conclude(note)
    reply = note.outcome("uncertain", evidence=["current view obscured"], evidence_step=7)
    assert reply["assessment"] is None and reply["phase"] == "retry_failed"
    entries = re.findall(r'```json\n(.*?)\n```', (tmp_path / note.record).read_text(), re.S)
    assert json.loads(entries[1])["question"] is None
    assert json.loads(entries[-1])["assessment"] is None


def test_replan_keeps_history_without_inventing_an_outcome(case, tmp_path):
    note, _, _ = case
    begin(note)
    classify(note)
    first = conclude(note)["planned_segment"]
    note.observe()
    second = note.conclude("Use a different approach", next_segment="Try new yaw",
                           continue_when="Target remains held")["planned_segment"]
    assert first != second and note._phase == "concluded"
    record = (tmp_path / note.record).read_text()
    assert record.count("## Revised diagnosis and proposed correction") == 2
    assert "## Retry outcome reported by agent" not in record


@pytest.mark.parametrize("step", [6, 7])
def test_stale_outcome_reports_rejection_without_interrupting_cell(case, tmp_path, capsys, step):
    note, robot, shown = case
    begin(note)
    classify(note)
    plan = conclude(note)["planned_segment"]
    before = (tmp_path / note.record).read_bytes()
    robot.state = lambda step=-1: State(step=8)
    reply = note.outcome("passed", evidence=["old holding"], evidence_step=step)
    assert reply["phase"] == "concluded" and note._segment == plan
    assert "not recorded" in reply["prompt"] and "8" in reply["prompt"]
    assert "not recorded" in capsys.readouterr().out
    assert (tmp_path / note.record).read_bytes() == before and not shown
    assert note.outcome("uncertain", evidence=["step 8 obscured"], evidence_step=8)["phase"] == "retry_failed"


def test_compact_replies_preserve_full_records_and_images(tmp_path):
    before = Path(__file__).with_name("fixtures") / "recovery_before_compaction.py"
    spec = importlib.util.spec_from_file_location("recovery_before_compaction", before)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    runs = []
    for label, cls in (("before", old.StaticRecovery), ("after", StaticRecovery)):
        folder = tmp_path / label
        folder.mkdir()
        shown = []
        robot = ReadOnlyRobot()
        note = cls(robot, open_file=lambda name, *a, **kw: open(folder / name, *a, **kw),
                   display_image=shown.append)
        replies = {"begin": begin(note)}
        replies["localize"] = note.localize(steps=[
            {"action": "lift", "status": "uncertain", "evidence": ["step 3: hold obscured"]},
        ])
        replies["review"] = note.review(question="When was the hold lost?", condition="target held", checkpoints=[
            {"step": 3, "action": "lift", "cameras": ["cam_head", "cam_left_wrist"]},
        ])
        replies["classify"] = classify(note)
        replies["observe"] = note.observe(cameras=["cam_head", "cam_left_wrist"],
            measurements={"locate": {"found": False, "note": "occluded", "views": 1}})
        if label == "before":
            replies["conclude"] = note.conclude(
                "Target offset", "stale target", evidence=["head image"], uncertainty="contact unknown",
                correction="Use the updated measured center for the next approach.",
                next_segment="Approach above the measured center; stop before descent.",
                return_evidence="Wrist image and actual pose after approach.",
                continue_when="Target is visibly centered and the descent path is clear.")
        else:
            replies["conclude"] = conclude(note)
        replies["retry_observe"] = note.observe(cameras=["cam_left_wrist"])
        replies["outcome"] = note.outcome("uncertain", evidence=["step 7: hold still obscured"],
            evidence_step=7, assessment="Current view cannot confirm hold; no resolving evidence.")
        runs.append((replies, {p.name: p.read_bytes() for p in folder.iterdir()}, shown, robot.calls))
    previous, current = runs
    assert previous[2:] == current[2:]
    assert {k: v for k, v in previous[1].items() if k.endswith('.png')} == {k: v for k, v in current[1].items() if k.endswith('.png')}
    old_entries, new_entries = [
        [json.loads(v) for v in re.findall(r'```json\n(.*?)\n```', run[1]['failure_001.md'].decode(), re.S)]
        for run in (previous, current)]
    for old_entry, new_entry in zip(old_entries, new_entries, strict=True):
        if 'proposed_correction' in new_entry:
            assert new_entry['proposed_correction'] == old_entry['proposed_correction']
            assert new_entry['start_step'] == old_entry['start_step']
            assert not {'findings', 'revised_hypothesis', 'remaining_uncertainty', 'return_evidence'} & new_entry.keys()
        elif 'planned_segment' in new_entry:
            old_entry['planned_segment'].pop('return_evidence')
            assert old_entry == new_entry
        else:
            assert old_entry == new_entry
    assert sum(len(str(v)) for v in current[0].values()) < sum(len(str(v)) for v in previous[0].values())
    for name, reply in current[0].items():
        assert ("scope" in reply) == (name == "localize")
        assert ("recovery_goal" in reply) == (name in ("begin", "conclude", "retry_observe", "outcome"))
    for name in ("observe", "retry_observe"):
        for key in ("images", "state", "recorded_step", "measurements"):
            assert current[0][name][key] == previous[0][name][key]
    assert current[0]["review"]["checkpoints"] == previous[0]["review"]["checkpoints"]
    assert current[0]["outcome"]["result"] == "uncertain"
    assert current[0]["outcome"]["assessment"] == "Current view cannot confirm hold; no resolving evidence."
