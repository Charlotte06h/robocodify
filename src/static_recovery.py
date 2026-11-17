"""Experimental recovery evidence for PyRUA-Lean's existing Python cells.

Copy this file into the episode workspace. Pass that cell's guarded ``open``
and ``display`` to StaticRecovery; all robot access uses public read-only APIs.
The agent still chooses and executes any observation move through ``robo``.
"""

import json
from dataclasses import asdict, is_dataclass

import numpy as np


LOCALIZE_PROMPT = "Call localize(steps=...) to record the unresolved action chain."
CLASSIFY_PROMPT = "Classify the unresolved effect; use unknown when the cause is unclear."
BOUNDARY_PROMPT = "Use review for unclear historical loss, observe for missing current evidence. "
BOUNDARY_REVIEW_PROMPT = "Identify the last established and first violated boundary, then update the diagnosis; cause may remain unknown."
OBSERVE_PROMPT = "Answer the question with existing evidence or observe; reuse the original Perception guidance."
REVIEW_PROMPT = "Use the evidence to choose a meaningful change to try, or collect a missing view."
RETRY_PROMPT = "Use the recorded revision variables in the actual retry; check original motion feedback and return for needed model interpretation."
OUTCOME_PROMPT = "Assess continue_when and recovery_goal from current evidence, then record outcome."


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
        if isinstance(item, (np.ndarray, np.generic)):
            return item.tolist()
        raise TypeError(f"unsupported evidence value: {type(item).__name__}")

    return json.dumps(value, default=default, ensure_ascii=False, indent=2, allow_nan=False)


def _cameras(cameras):
    allowed = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    if (not isinstance(cameras, (list, tuple))
            or any(camera not in allowed for camera in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError("cameras must be a unique selection of RoboDojo cameras")
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
        # Cells echo only their final expression; assigned replies still need guidance.
        print(f"{self.record}: {prompt}")
        return {
            "record": self.record,
            "phase": self._phase,
            "prompt": prompt,
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
        After passed, begin starts a separate record on the same instance.
        """
        self._require(None, "checked")
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
        self._observation = self._review = 0
        self._question = self._error_type = self._scope = self._segment = None
        self.record = path
        self._recovery_goal = data["recovery_goal"]
        self._phase = "reported"
        return self._reply(LOCALIZE_PROMPT, recovery_goal=self._recovery_goal)

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
        return self._reply(BOUNDARY_PROMPT + CLASSIFY_PROMPT, scope=scope)

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
            if not point["cameras"]:
                raise ValueError("review checkpoints need at least one camera")
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

    def classify(self, error_type, hypothesis, *, evidence, question=None):
        """Append a provisional cause; an omitted question is recorded as unspecified."""
        self._require("reported", "localized", "classified", "observed", "concluded", "retry_failed", "progress")
        data = {
            "source": "agent judgment, not a verified physical cause",
            "error_type": _text(error_type, "error_type"),
            "hypothesis": _text(hypothesis, "hypothesis"),
            "evidence": _evidence(evidence),
            "question": None if question is None else _text(question, "question"),
        }
        self._append("Provisional diagnosis", data)
        self._question = data["question"]
        self._error_type = data["error_type"]
        self._segment = None
        self._phase = "classified"
        return self._reply(OBSERVE_PROMPT, error_type=self._error_type, question=self._question)

    def observe(self, *, cameras=("cam_head",), measurements=None):
        """Save and display current static views at one recorded step.

        Args:
            cameras: selection of existing cameras; empty saves only state/measurements.
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
            **({"recovery_goal": self._recovery_goal, "planned_segment": self._segment}
               if retry else {"question": self._question}),
        )

    def conclude(self, correction, *, next_segment, continue_when):
        """Record a change to try and its check; never accept a task stage.

        May use evidence already in context without a new observe() call.
        Images collected in this cell are only read by the model after cell return.
        correction includes the reason or hypothesis being tested. continue_when
        names the condition and evidence to check, not a prerequisite for trying.
        Repeating conclude replaces the pending plan and preserves its recorded history.
        This is guidance, not a runtime motion guard.
        """
        self._require("classified", "observed", "progress", "concluded")
        segment = {
            "next_segment": _text(next_segment, "next_segment"),
            "continue_when": _text(continue_when, "continue_when"),
        }
        data = {
            "source": "agent-proposed trial; cause may remain unknown",
            "proposed_correction": _text(correction, "correction"),
            "recovery_goal": self._recovery_goal,
            **segment,
        }
        segment["start_step"] = self.robo.state().step
        data["start_step"] = segment["start_step"]
        self._append("Revised diagnosis and proposed correction", data)
        self._phase = "concluded"
        self._segment = segment
        return self._reply(RETRY_PROMPT, recovery_goal=self._recovery_goal, planned_segment=segment)

    def outcome(self, result, *, evidence, evidence_step, assessment=None):
        """Record the agent's current-evidence judgment, not a task-success oracle.

        evidence_step identifies the inspected post-segment record. Optional assessment
        explains support and contrary evidence; its meaning is not verified here.
        Stale evidence prints guidance without recording an outcome or stopping the cell.
        """
        self._require("concluded")
        if result not in ("progress", "passed", "failed", "uncertain"):
            raise ValueError("result must be progress, passed, failed or uncertain")
        evidence = _evidence(evidence)
        assessment = None if assessment is None else _text(assessment, "assessment")
        if type(evidence_step) is not int or evidence_step < 0:
            raise ValueError("evidence_step must be a nonnegative recorded index")
        state = self.robo.state()
        if evidence_step > state.step:
            raise ValueError("evidence_step cannot be in the future")
        if evidence_step < state.step or evidence_step < self._segment["start_step"]:
            return self._reply(
                f"Outcome not recorded: inspect current evidence at recorded step {state.step} "
                "before outcome; do not relabel earlier evidence as current.",
                evidence_step=evidence_step,
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
            "Record closed; continue with the original task checks."
            if result == "passed"
            else "Goal remains unmet; conclude the next segment, or revisit diagnosis if needed."
            if result == "progress"
            else "Reclassify before observe/conclude; review or re-localize first if needed."
        )
        return self._reply(prompt, result=result, evidence_step=evidence_step,
                           assessment=assessment, recovery_goal=self._recovery_goal)
