"""Verify the packaging hook without starting a robot or model service."""

import hashlib
import importlib.util
import json
from pathlib import Path

from pyrualean import play


def test_launcher_preserves_author_prompt_and_arguments(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "recovery_launcher", root / "integrations/pyrua_lean/run.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.syspath_prepend(str(workspace))
    baseline = "Author task and perception guidance.\n"
    original = lambda *args, **kwargs: baseline
    monkeypatch.setattr(play, "build_prompt", original)
    arguments = ["--arm", "cells", "--task", "build_tower"]

    def run(argv):
        assert argv is arguments
        text = play.build_prompt("cells", object(), budget_s=5400, max_programs=None)
        assert text == baseline + "\n\n" + launcher.PROMPT.read_text(encoding="utf-8").strip() + "\n"
        assert (workspace / "static_recovery.py").read_bytes() == launcher.MODULE.read_bytes()
        assert (tmp_path / "baseline-prompt.txt").read_text() == baseline
        record = json.loads((tmp_path / "static-recovery-installation.json").read_text())
        assert record["baseline_prompt_sha256"] == hashlib.sha256(baseline.encode()).hexdigest()
        assert record["method_prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        return 0

    monkeypatch.setattr(play, "main", run)
    assert launcher.main(arguments) == 0
    assert play.build_prompt is original
