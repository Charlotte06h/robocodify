"""Experimental recovery evidence for PyRUA-Lean's existing Python cells.

Copy this file into the episode workspace. Pass that cell's guarded ``open``
and ``display`` to StaticRecovery; all robot access uses public read-only APIs.
The agent still chooses and executes any observation move through ``robo``.
"""

import json
from dataclasses import asdict, is_dataclass


LOCALIZE_PROMPT = """Call localize(steps=[...]) with the short,
ordered chain of dependent actions around this failure. Each step has action,
status (passed, failed, uncertain or not_attempted), and evidence references;
only not_attempted may have empty evidence. Mark unverified effects uncertain.
A past passed step does not establish that its result still holds now.
Use the returned scope; continue diagnosis and planning in the same cell when
existing evidence or in-code checks suffice. Recording calls do not require a new turn."""

CLASSIFY_PROMPT = """Classify the reported failure using the recorded evidence.
Separate the observed symptom from a possible cause; an exception, unreached
motion or closed gripper alone does not establish an object-level failure.
State the provisional error type, hypothesis, supporting evidence references,
and the current fact that must be checked. Use 'unknown' when necessary.
If a local scope is recorded, focus on its first unresolved action; check which
earlier conditions still hold. If that action combines effects that need separate
confirmation, identify the first condition to check rather than retrying the whole chain.
If no unresolved action was reported, revisit the failure evidence rather than infer success.
Call classify(...) to record that judgment. It may share a cell with localization,
observation planning and a correction supported by existing evidence."""

BOUNDARY_PROMPT = """Before assigning a cause, check what evidence is missing.
For an unclear loss of a previously established condition, call review(question=...,
condition=..., checkpoints=[{'step': ..., 'action': ..., 'cameras': (...)}]) with
selected recorded action boundaries. Return images before any judgment that
requires the main agent to inspect them.
Use robo.state().step indices, NOT Move.step or the native action count; state(step=k)
and artifacts refer to these recorded boundaries. Action labels are your references,
not verified by this program. Start with the last confirmed condition and boundaries
of intervening actions that may change it; do not review the whole episode.
Choose wrist views for contact/holding, head for layout/arm interference; add another
view where visibility is uncertain. For a rejected command with no motion or a
current geometry question, existing feedback or static observation may suffice.
If historical evidence is missing, preserve uncertainty and plan a check on retry.
Otherwise follow the classification guidance below.
"""

BOUNDARY_REVIEW_PROMPT = """Inspect these historical images with their recorded poses
and the stated condition. For each boundary say established, violated or unclear,
citing step and camera. Identify the last established and first violated boundary;
unclear views do not establish absence. Distinguish a loss interval from its physical
cause: endpoints alone may not prove slip or collision. If needed, request another
view or available boundary within that interval, not a full-video review.
After inspecting the images, update localize(...) if the evidence changes the failure scope,
or classify(...) with these findings and remaining uncertainty. Do not prewrite
those judgments in the cell collecting images. Use the existing observe/conclude
flow in the same cell when the inspected evidence supports it; in correction
name the condition to recheck before dependent retry actions.
Historical frames do not establish the current scene or certify task success."""

OBSERVE_PROMPT = """Use the recorded question to choose a static observation.
First inspect available images, robot state and motion results. If they already
answer it, explain which evidence answers the question in conclude(...); no
camera move is needed. Do not treat an occluded or uninformative view as confirmation.
Otherwise reuse the original Perception guidance: locate for an initial estimate,
move an available wrist to a useful view, then inspect and measure again. For
low-object geometry, consider the original above-object view; tilt where needed
for reach. For contact or support relations, choose a view that exposes that
relation rather than always moving above it. Derive targets from the current
scene, keep necessary holds, and consider objects and the other arm along the
move. Use the existing move_to and its feedback; it does not plan full scene
collision avoidance. Check actual visibility and locate's views/note, not just
reached. Call observe(...) with the relevant cameras and measurement outputs.
If the next decision requires reading those images, return them and inspect them
next turn. Otherwise existing evidence and in-code checks may support continuing
in the same cell; do not prewrite judgments about unseen images."""

REVIEW_PROMPT = """Inspect these static images together with their measured state
and the recorded question; camera calibration is in the note if needed.
Check object identity, visibility
and measurement assumptions. A changed measurement is not proof of improvement;
locate may be unreliable for held or stacked objects. Call conclude(...) with
what the evidence establishes, what remains unknown, the revised cause hypothesis,
and one meaningful correction with its evidence references. State which evidence
invalidates the previous plan, what changes, and how return_evidence will test
that change; reuse the original effect checks. Another static look
is allowed when a necessary fact remains missing. In correction, split only the
local actions whose next parameters depend on new evidence; name the next effect
to achieve and where new evidence requires a model decision. Call conclude(...)
with next_segment, return_evidence and continue_when, then execute the supported
correction in the same cell. These frames do not show the
earlier failure process; do not infer a slip or collision history from them alone."""

RETRY_PROMPT = """Execute the recorded next_segment using the original tools
and checks, in the same cell as conclude when supported. Collect return_evidence.
When continue_when requires the main agent to interpret new images or other
feedback, return that evidence before dependent actions. Otherwise evaluate it
in code, record outcome and plan/execute the next supported segment in the same cell.
Do not prewrite judgments about unseen images. This program
records the boundary; it does not stop a cell or execute actions automatically.
Record outcome('progress', evidence=[...], evidence_step=..., assessment=...) for an achieved intermediate effect
while recovery_goal remains unmet; clearing a view or remeasuring is not recovery
success. Use passed only when evidence supports the fixed recovery_goal in full.
Failed or uncertain checks stay in recovery. Do not repeat a failed action without
a reason or infer a cause beyond the evidence. This record is not task success."""


OUTCOME_PROMPT = """Judge continue_when and recovery_goal from evidence AFTER the
segment, using the original Judging progress guidance and tool checks. Call
outcome(result, evidence=[...], evidence_step=..., assessment=...). evidence_step
is the recorded robo.state().step of the evidence inspected, not a native action
count; it must still be current. In assessment state what is established or unmet,
address contrary evidence (or say none observed), and explain any resolution.
Earlier reached/holding results do not establish a later object condition. If
evidence conflicts or is unclear, use uncertain unless it establishes failure;
inspect missing evidence before continuing. Reuse available current images and
state; no new camera move or capture is required when they suffice. This program
checks the record index, not whether your interpretation is correct."""


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _evidence(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("evidence must contain observed facts and their references")
    return [_text(item, "evidence item") for item in value]


def _json(value):
    def default(item):
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        raise TypeError(f"unsupported evidence value: {type(item).__name__}")

    return json.dumps(value, default=default, ensure_ascii=False, indent=2, allow_nan=False)


def _cameras(cameras):
    allowed = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    if (not isinstance(cameras, (list, tuple)) or not cameras
            or any(camera not in allowed for camera in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError("cameras must be a non-empty, unique selection of RoboDojo cameras")
    return cameras


def _poses(state):
    state_dict = json.loads(_json(state))
    keys = ("tcp_pos", "pitch", "yaw", "roll", "gripper")
    return {arm: {key: state_dict[arm][key] for key in keys} for arm in ("left", "right")}


class StaticRecovery:
    """One failure's append-only Markdown record; judgments remain the agent's.

    Args:
        robo: the existing public robot object.
        open_file: the current cell's guarded ``open``.
        display_image: the current cell's ``display`` (accepts PNG bytes).
    """

    def __init__(self, robo, *, open_file, display_image):
        self.robo = robo
        self.open_file = open_file
        self.display_image = display_image
        self.record = None
        self._phase = None
        self._observation = 0
        self._review = 0
        self._question = None
        self._error_type = None
        self._scope = None
        self._recovery_goal = None
        self._segment = None

    def _require(self, *phases):
        if self._phase not in phases:
            raise ValueError(f"call not valid after {self._phase!r}; expected {phases}")

    def _entry(self, title, data):
        return f"\n## {title}\n\n```json\n{_json(data)}\n```\n"

    def _append(self, title, data):
        entry = self._entry(title, data)
        with self.open_file(self.record, "a", encoding="utf-8") as stream:
            stream.write(entry)

    def _reply(self, prompt, **extra):
        return {
            "record": self.record,
            "phase": self._phase,
            "error_type": self._error_type,
            "question": self._question,
            "recovery_goal": self._recovery_goal,
            "prompt": prompt,
            **({"scope": self._scope} if self._scope is not None else {}),
            **({"planned_segment": self._segment} if self._segment is not None else {}),
            **extra,
        }

    def _capture(self, step, cameras, prefix):
        frames = []
        for camera in cameras:
            png = self.robo.artifact(f"{camera}.png", step=step)
            meta = self.robo.camera_meta(camera, step=step)
            path = f"{prefix}_{camera}.png"
            with self.open_file(path, "xb") as stream:
                stream.write(png)
            frames.append((camera, path, meta, png))
        return frames

    def begin(self, stage, failure, *, recovery_goal, evidence):
        """Record an agent-reported failure and request its local action scope.

        No failure is inferred automatically. Evidence names the original cell,
        tool result or image and what was actually observed there.
        recovery_goal fixes the condition to restore, not an intermediate repair.
        """
        self._require(None)
        data = {
            "source": "agent-reported failure; not independently verified",
            "stage": _text(stage, "stage"),
            "failure": _text(failure, "failure"),
            "recovery_goal": _text(recovery_goal, "recovery_goal"),
            "evidence": _evidence(evidence),
            "state_at_registration": self.robo.state(),
        }
        body = "# Static recovery record (experimental)\n"
        body += "\nAgent judgments are provisional; captured state and images are separate evidence.\n"
        body += self._entry("Reported failure", data)
        index = 1
        while True:
            path = f"failure_{index:03d}.md"
            try:
                with self.open_file(path, "x", encoding="utf-8") as stream:
                    stream.write(body)
                break
            except FileExistsError:
                index += 1
        self.record = path
        self._recovery_goal = data["recovery_goal"]
        self._phase = "reported"
        return self._reply(LOCALIZE_PROMPT)

    def localize(self, *, steps):
        """Record a local action chain and locate its first reported unresolved effect.

        Status and evidence are agent judgments, not independently verified.
        A passed step is historical evidence, not a guarantee of its current state.
        This does not execute, schedule or accept any task action.
        """
        self._require("reported", "localized", "classified", "observed", "concluded", "retry_failed", "progress")
        if not isinstance(steps, (list, tuple)) or not steps:
            raise ValueError("steps must be a non-empty ordered list")
        recorded = []
        for step in steps:
            if not isinstance(step, dict) or set(step) != {"action", "status", "evidence"}:
                raise ValueError("each step needs action, status and evidence")
            status = step["status"]
            if status not in ("passed", "failed", "uncertain", "not_attempted"):
                raise ValueError("step status must be passed, failed, uncertain or not_attempted")
            evidence = step["evidence"]
            recorded.append({
                "action": _text(step["action"], "action"),
                "status": status,
                "evidence": [] if status == "not_attempted" and evidence in ([], ()) else _evidence(evidence),
            })
        first = next((i for i, step in enumerate(recorded) if step["status"] in ("failed", "uncertain")), None)
        last = next((i for i in range((first if first is not None else len(recorded)) - 1, -1, -1)
                     if recorded[i]["status"] == "passed"), None)
        scope = {
            "last_confirmed": None if last is None else {"step": last + 1, **recorded[last]},
            "first_unresolved": None if first is None else {"step": first + 1, **recorded[first]},
            "not_attempted": [i + 1 for i, step in enumerate(recorded) if step["status"] == "not_attempted"],
        }
        self._append("Local action scope", {
            "source": "agent-reported action statuses; not a verified physical cause",
            "steps": recorded,
            **scope,
        })
        self._scope = scope
        self._question = self._error_type = None
        self._segment = None
        self._phase = "localized"
        return self._reply(BOUNDARY_PROMPT + CLASSIFY_PROMPT)

    def review(self, *, question, condition, checkpoints):
        """Display selected historical action boundaries using public recorded reads.

        Each checkpoint supplies step (nonnegative recorded index), action (agent
        reference) and cameras. Images and poses come from that exact step; labels
        and subsequent visual judgments remain the agent's. No motion or model call.
        Missing evidence raises and is recorded, never replaced by a current image.
        """
        self._require("localized", "classified", "observed", "progress", "retry_failed")
        question, condition = _text(question, "question"), _text(condition, "condition")
        if not isinstance(checkpoints, (list, tuple)) or not checkpoints:
            raise ValueError("checkpoints must be a non-empty list")
        selected = []
        for point in checkpoints:
            if not isinstance(point, dict) or set(point) != {"step", "action", "cameras"}:
                raise ValueError("each checkpoint needs step, action and cameras")
            step = point["step"]
            if type(step) is not int or step < 0:
                raise ValueError("step must be a nonnegative recorded index")
            selected.append({"step": step, "action": _text(point["action"], "action"),
                             "cameras": list(_cameras(point["cameras"]))})
        if len({point["step"] for point in selected}) != len(selected):
            raise ValueError("combine cameras for the same recorded step")
        selected.sort(key=lambda point: point["step"])
        self._review += 1
        records, images, summaries = [], [], []
        try:
            for point in selected:
                step = point["step"]
                state = self.robo.state(step=step)
                if state.step != step:
                    raise ValueError(f"historical state does not match requested step {step}")
                prefix = f"{self.record[:-3]}_review_{self._review:03d}_step_{step:04d}"
                frames = self._capture(step, point["cameras"], prefix)
                records.append({"step": step, "action_reference": point["action"], "state": state,
                                "frames": [{"camera": cam, "file": path, "calibration": meta}
                                           for cam, path, meta, _ in frames]})
                summaries.append({"step": step, "action_reference": point["action"],
                                  "state": _poses(state), "images": [path for _, path, _, _ in frames]})
                images.extend((step, cam, path, png) for cam, path, _, png in frames)
            self._append(f"Boundary review {self._review}", {
                "source": "public historical reads; action labels and question supplied by agent",
                "question": question, "condition": condition, "checkpoints": records,
            })
            with self.open_file(self.record, "a", encoding="utf-8") as stream:
                for step, camera, path, _ in images:
                    stream.write(f"\n![{camera}, recorded step {step}]({path})\n")
            for _, _, _, png in images:
                self.display_image(png)
        except Exception as exc:
            try:
                self._append("Boundary review error", {
                    "review": self._review, "question": question, "condition": condition,
                    "requested": selected, "error": f"{type(exc).__name__}: {exc}",
                })
            except Exception:
                raise exc
            raise
        return self._reply(BOUNDARY_REVIEW_PROMPT, question=question, condition=condition,
                           checkpoints=summaries)

    def classify(self, error_type, hypothesis, *, evidence, question):
        """Append a provisional cause and return the original-style view guidance."""
        self._require("reported", "localized", "classified", "observed", "concluded", "retry_failed", "progress")
        data = {
            "source": "agent judgment, not a verified physical cause",
            "error_type": _text(error_type, "error_type"),
            "hypothesis": _text(hypothesis, "hypothesis"),
            "evidence": _evidence(evidence),
            "question": _text(question, "question"),
        }
        self._append("Provisional diagnosis", data)
        self._question = data["question"]
        self._error_type = data["error_type"]
        self._segment = None
        self._phase = "classified"
        return self._reply(OBSERVE_PROMPT)

    def observe(self, *, cameras=("cam_head",), measurements=None):
        """Save and display current static views at one recorded step.

        Args:
            cameras: non-empty selection of the three existing RoboDojo cameras.
            measurements: named outputs already obtained from original tools.
                These are agent-supplied, not re-run or assigned a new timestamp.
        Returns:
            Compact state, file references, measurements and a review prompt.
            PNG bytes are also sent to the existing display callback.
            During a planned retry, preserve its segment for outcome(...).
        Raises:
            ValueError: malformed arguments or wrong diagnostic phase.
            Exception: an original read/write/display error, after recording it
                when possible. No replacement image or robot motion is attempted.
        """
        self._require("classified", "observed", "progress", "concluded")
        _cameras(cameras)
        if measurements is not None and not isinstance(measurements, dict):
            raise ValueError("measurements must be a dictionary of named tool outputs")
        # Serialize now so later mutation of a tool result cannot change the note.
        measured = json.loads(_json(measurements or {}))
        self._observation += 1
        prefix = f"{self.record[:-3]}_observation_{self._observation:03d}"
        try:
            state = self.robo.state()
            step = state.step
            frames = self._capture(step, cameras, prefix)
            data = {
                "source": "public robot reads at the recorded step",
                "question": self._question,
                "recorded_step": step,
                "state": state,
                "frames": [
                    {"camera": cam, "file": path, "calibration": meta}
                    for cam, path, meta, _ in frames
                ],
                "agent_supplied_measurements": measured,
            }
            self._append(f"Static observation {self._observation}", data)
            with self.open_file(self.record, "a", encoding="utf-8") as stream:
                for camera, path, _, _ in frames:
                    stream.write(f"\n![{camera}, recorded step {step}]({path})\n")
            for _, _, _, png in frames:
                self.display_image(png)
        except Exception as exc:
            try:
                self._append(
                    "Observation error",
                    {"observation": self._observation, "error": f"{type(exc).__name__}: {exc}"},
                )
            except Exception:
                raise exc
            raise
        retry = self._phase == "concluded"
        if not retry:
            self._phase = "observed"
        return self._reply(
            OUTCOME_PROMPT if retry else REVIEW_PROMPT,
            recorded_step=step,
            state=_poses(state),
            images=[path for _, path, _, _ in frames],
            measurements=measured,
        )

    def conclude(self, findings, hypothesis, *, evidence, uncertainty, correction,
                 next_segment, return_evidence, continue_when):
        """Append the agent's evidence-based revision; never accept a task stage.

        May use evidence already in context without a new observe() call.
        Images collected in this cell are only read by the model after cell return.
        The segment names its action boundary, evidence to return and condition
        to judge before continuing. It is guidance, not a runtime motion guard.
        """
        self._require("classified", "observed", "progress")
        segment = {
            "next_segment": _text(next_segment, "next_segment"),
            "return_evidence": _text(return_evidence, "return_evidence"),
            "continue_when": _text(continue_when, "continue_when"),
        }
        data = {
            "source": "agent judgment after inspecting evidence",
            "findings": _text(findings, "findings"),
            "revised_hypothesis": _text(hypothesis, "hypothesis"),
            "evidence": _evidence(evidence),
            "remaining_uncertainty": _text(uncertainty, "uncertainty"),
            "proposed_correction": _text(correction, "correction"),
            "recovery_goal": self._recovery_goal,
            **segment,
        }
        segment["start_step"] = self.robo.state().step
        data["start_step"] = segment["start_step"]
        self._append("Revised diagnosis and proposed correction", data)
        self._phase = "concluded"
        self._segment = segment
        return self._reply(RETRY_PROMPT + "\n" + OUTCOME_PROMPT)

    def outcome(self, result, *, evidence, evidence_step, assessment):
        """Record the agent's current-evidence judgment, not a task-success oracle.

        evidence_step identifies the inspected post-segment record. assessment
        explains support and contrary evidence; its meaning is not verified here.
        """
        self._require("concluded")
        if result not in ("progress", "passed", "failed", "uncertain"):
            raise ValueError("result must be progress, passed, failed or uncertain")
        evidence = _evidence(evidence)
        assessment = _text(assessment, "assessment")
        if type(evidence_step) is not int or evidence_step < 0:
            raise ValueError("evidence_step must be a nonnegative recorded index")
        state = self.robo.state()
        if evidence_step != state.step or evidence_step < self._segment["start_step"]:
            raise ValueError(
                f"inspect evidence at current recorded step {state.step} before outcome; "
                "do not relabel earlier evidence as current"
            )
        self._append(
            "Retry outcome reported by agent",
            {"result": result, "evidence": evidence, "evidence_step": evidence_step,
             "assessment": assessment, "state_at_judgment": state,
             "recovery_goal": self._recovery_goal, "planned_segment": self._segment},
        )
        self._phase = "checked" if result == "passed" else "progress" if result == "progress" else "retry_failed"
        self._segment = None
        prompt = (
            "This recovery record is closed on your evidence for recovery_goal. Do not submit "
            "outcome again; use a new case for a later failure. Use the original task checks "
            "to decide what follows; this record is not task success."
            if result == "passed"
            else "Intermediate progress only; recovery_goal is unchanged and this case remains open. "
            "Use the returned evidence to plan the next segment with conclude(...); no new failure "
            "diagnosis is needed if the plan remains supported. If a condition failed or is unclear, "
            "use localize/classify and observe/review as needed. Do not resume the normal plan or "
            "mark passed merely because the intermediate action succeeded."
            if result == "progress"
            else "Use review(...) first if historical boundaries are needed. "
            "Call classify(...) again before observe(...) or conclude(...), using this "
            "record to reconsider the cause before another correction. If the failure location "
            "changed or an earlier result was lost, call localize(...) first. Static views "
            "may not explain the process; preserve uncertainty and use the original recovery rules. "
            "In conclude, connect the changed action to the evidence that invalidated the old plan "
            "and specify how the next segment will test the change."
        )
        return self._reply(prompt)
