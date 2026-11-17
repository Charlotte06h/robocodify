## Static recovery (our experimental appendix)
Use the original Perception, Judging progress, Safety and recovery guidance and
motion checks. On failure, use `StaticRecovery` from `static_recovery.py` to keep
a replayable record and revise the local plan. It does not detect failure, move
the robot or certify success. Combine recording, planning and execution in one
cell; return when a decision needs model interpretation of new images/feedback.
Numeric checks may branch in code; do not prewrite judgments of unseen images.

- Create `case = StaticRecovery(robo, open_file=open, display_image=display)`.
  `case.begin(stage, failure, recovery_goal=..., evidence=[...])` fixes the goal.
  Reuse the case until resolved; after passed, begin opens a new record on it.
  Keep only facts needed to revisit the failure. evidence lists text references;
  pass raw tool objects through observe's measurements dictionary.
- `case.localize(steps=[{'action': ..., 'status': ..., 'evidence': [...]}])`
  identifies the first unresolved effect. Status: passed/failed/uncertain/
  not_attempted; only not_attempted may have empty evidence. Revisit this short
  chain if earlier work was lost; past success does not establish current state.
- `case.classify(error_type, hypothesis, evidence=[...], question=...)` records
  a provisional cause and the question to resolve. Unknown is valid. Change a
  meaningful variable and try using the original recovery guidance: a trial can
  test a hypothesis and produce new information without first proving the cause.
  question is optional when no new measurement question is needed.
- Reuse available evidence, or use original perception/motion tools to acquire
  missing information. `obs = case.observe(cameras=(...), measurements={...})`
  saves and displays current images/poses. Pass actual measurement objects,
  executed argument dictionaries and tool receipts through measurements, rather
  than retyping values. They are snapshotted as supplied, not remeasured or dated
  anew. Use wrist/head views appropriate to the question.
  NumPy values are supported; cameras=() records only poses/measurements.
- For unclear historical loss, `case.review(question=..., condition=...,
  checkpoints=[{'step': ..., 'action': ..., 'cameras': (...)}])` displays selected
  action boundaries with recorded poses. Use recorded `robo.state().step`
  indices, not cell numbers or native action counts. A loss interval is not proof
  of its physical cause; occluded evidence may remain uncertain.
- `case.conclude(correction, next_segment=..., continue_when=...)` records what
  to change and why, the next actions, and their check including needed evidence.
  Reuse diagnosis already recorded; no repeated justification is needed. Keep the
  change in Python variables used by the actual retry, including after remeasurement.
  You may revise a pending plan with conclude; this does not mark it successful.
  Execute with original tools and judge feedback/object response with original
  checks. During recovery, if a later action needs a measurement or visual judgment
  available only after an earlier action, execute up to that observation point.
  Handle numeric feedback in the same cell; return only when interpreting new images
  requires the agent. Recompute dependent targets from the new evidence. Otherwise
  batch the retry; uncertainty alone does not require another check.
- `case.outcome(result, evidence=[...], evidence_step=..., assessment=...)`
  records the post-action judgment against continue_when and the fixed goal.
  assessment is optional when the evidence already explains the judgment.
  For inspected observe images use `evidence_step=obs['recorded_step']`; for other
  evidence retain its original recorded index. It must still be current: an old
  reached/holding signal does not establish later success. The program checks
  indices, not visual correctness. Use progress for partial recovery, passed for
  the full goal, failed for failure, uncertain when unresolved. Progress permits
  conclude; failed/uncertain requires classify before observe/conclude, with
  review/localize if needed. Passed closes the record. No retry is automatic.
  Stale evidence leaves the outcome unrecorded and prints guidance without stopping
  the cell or taking new images; inspect current evidence before recording it again.

Each call prints its short guidance; do not print its full reply unless needed.
Print needed evidence and the final reply. The program
maintains record paths, phases and evidence indices. Full records/images remain
in `case.record`; read relevant entries only when needed for the next decision.
