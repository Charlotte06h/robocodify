# Project working rules

- Preserve the pinned PyRUA-Lean baseline. Separate our method changes from
  environment adaptation; do not silently change prompts, tools, budgets or scoring.
- Prefer subtraction and existing helpers. Keep the recovery module and prompt
  small; normal actions retain the author's execution/checking behavior.
- Run `python -m pytest -q` after behavioral changes. The pinned upstream checkout
  and test setup are documented in README; never replace a missing dependency by
  silently skipping the integration test.
- Keep README as the entrypoint, docs/implementation.md as current behavior,
  docs/research_ideas.md as research intentions, and experiments/ as evidence indices.
  Update existing documents rather than adding a summary for each task.
- Never commit credentials, machine-specific private configuration, environment
  directories, downloaded assets, or full run logs. Keep frozen evidence externally.
- Preserve failures, interruptions and old configurations. Native results and
  Agent self-assessments are different fields; unverified causes remain hypotheses.
- Before server work, inspect existing jobs. Use named tmux sessions and logs for
  new long jobs; do not restart an existing run or use shared GPU resources implicitly.
- Before an experiment, freeze the commit and configuration. Repository cleanup
  does not authorize new robot experiments, model calls, or deleting old evidence.
