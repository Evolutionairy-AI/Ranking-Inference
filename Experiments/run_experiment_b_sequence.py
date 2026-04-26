"""Sequential orchestrator for Experiment B scoring.

FRANK is already running in the background (launched via the Bash tool).
This script waits for the FRANK output file to stabilise, then runs
TruthfulQA and HaluEval back-to-back so none of them contend for Ollama.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
VENV_PY = str(EXP_DIR / ".venv" / "Scripts" / "python.exe")

FRANK_OUTPUT = EXP_DIR / "exp06_frank" / "output" / "scored_frank_llama-3.1-8b.jsonl"
TRUTHFULQA_LOG = EXP_DIR / "exp05_truthfulqa" / "scoring_experiment_b.log"
HALUEVAL_LOG = EXP_DIR / "exp04_halueval" / "scoring_experiment_b.log"


def wait_for_stable(path: Path, stable_secs: int = 300,
                    poll_secs: int = 60, timeout_hours: float = 8) -> bool:
    """Return True once *path* has not changed in *stable_secs*.

    Polls every *poll_secs*; times out after *timeout_hours*.
    """
    deadline = time.time() + timeout_hours * 3600
    last_size = -1
    last_change = time.time()

    while time.time() < deadline:
        size = path.stat().st_size if path.exists() else 0
        if size != last_size:
            last_size = size
            last_change = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] {path.name}: {size:,} bytes (still growing)")
        elif size > 0 and (time.time() - last_change) > stable_secs:
            print(f"[{time.strftime('%H:%M:%S')}] {path.name} stable at {size:,} bytes for {stable_secs}s -- treating as done")
            return True
        time.sleep(poll_secs)
    print(f"[{time.strftime('%H:%M:%S')}] timed out waiting for {path}")
    return False


def run_cmd(cmd: list[str], log_path: Path) -> int:
    print(f"\n[{time.strftime('%H:%M:%S')}] launching: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # HuggingFace hub has been returning 503/504 intermittently, adding ~30s
    # per iteration via the tokenizer's HEAD check.  The tokenizer is cached
    # locally, so offline mode is safe and fast.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(EXP_DIR), stdout=log,
                                 stderr=subprocess.STDOUT, text=True,
                                 env=env)
        rc = proc.wait()
    print(f"[{time.strftime('%H:%M:%S')}] finished rc={rc}, log={log_path}")
    return rc


def main() -> int:
    print(f"[{time.strftime('%H:%M:%S')}] waiting for FRANK output to stabilise...")
    ok = wait_for_stable(FRANK_OUTPUT, stable_secs=300, poll_secs=60, timeout_hours=6)
    if not ok:
        print("FRANK did not stabilise within timeout -- continuing anyway")

    # 1) TruthfulQA (full 817 questions)
    rc = run_cmd(
        [VENV_PY, str(EXP_DIR / "exp05_truthfulqa" / "run.py"),
         "--model", "llama-3.1-8b"],
        TRUTHFULQA_LOG,
    )
    if rc != 0:
        print(f"TruthfulQA returned rc={rc}; continuing to HaluEval anyway")

    # 2) HaluEval (subsampled 2000/task = 6000 total)
    rc = run_cmd(
        [VENV_PY, str(EXP_DIR / "exp04_halueval" / "run.py"),
         "--model", "llama-3.1-8b", "--task", "all",
         "--max-examples", "2000"],
        HALUEVAL_LOG,
    )
    if rc != 0:
        print(f"HaluEval returned rc={rc}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Experiment B sequence complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
