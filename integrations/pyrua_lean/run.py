"""Append recovery evidence support to the unchanged PyRUA-Lean cells runner.

Adapted from our frozen experiment launchers. Only machine/task configuration
has moved to the author's existing CLI; the workspace/prompt hook is retained.
Run this source checkout with the pinned PyRUA-Lean environment.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

from pyrualean import play

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/static_recovery.py"
PROMPT = Path(__file__).with_name("prompt.md")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    original_prompt = play.build_prompt

    def build_prompt(*args, **kwargs):
        baseline = original_prompt(*args, **kwargs)
        workspace = Path.cwd()
        out = workspace.parent
        (out / "baseline-prompt.txt").write_text(baseline, encoding="utf-8")
        shutil.copyfile(MODULE, workspace / "static_recovery.py")
        # The original cell guard allows imports from the episode workspace.
        sys.path.insert(0, str(workspace))
        appendix = PROMPT.read_text(encoding="utf-8").strip()
        text = baseline + "\n\n" + appendix + "\n"
        record = {
            "baseline_prompt_sha256": sha(baseline.encode()),
            "baseline_prompt_chars": len(baseline),
            "method_prompt_sha256": sha(text.encode()),
            "method_prompt_chars": len(text),
            "appendix_chars": len(text) - len(baseline),
            "module_sha256": sha((workspace / "static_recovery.py").read_bytes()),
            "hook": "workspace module installation, import path and initial prompt append only",
        }
        (out / "static-recovery-installation.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        return text

    play.build_prompt = build_prompt
    try:
        return play.main(argv)
    finally:
        play.build_prompt = original_prompt


if __name__ == "__main__":
    raise SystemExit(main())
