"""Run a multi-arm ablation grid: one git worktree per arm, N evaluator servers.

Every arm gets its own worktree. This is not tidiness -- it is the only thing
that stops arms reading each other's kernels. `results/` is untracked, so a
fresh worktree contains no sibling data at all. Permission globs cannot do this
job: measured, `external_directory: deny` does NOT block absolute paths, and
read-denies bind only relative to the session directory, so a sibling arm or
the primary checkout stays readable through an absolute path. See RUNBOOK.md.

Server assignment is by task. Two B300 evaluators are not necessarily timing
equivalent (one pair measured a 26% gap on an identical kernel), so absolute
kernel_time_ms cannot be pooled across them. Pinning a whole task to one server
keeps every within-task comparison on one clock; speedup ratios survive a split
regardless, because each arm times its own baseline on its own server.

Keys come from the environment and are never written here.

    export DEEPSEEK_BASE_URL DEEPSEEK_API_KEY
    export GEMMA_BASE_URL GEMMA_API_KEY QWEN_BASE_URL QWEN_API_KEY
    export CUTE_HARNESS_API_KEY
    python experiments/cute_ablation/run_grid.py --plan plan.json --logs <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import string
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REQUIRED_ENV = ("CUTE_HARNESS_API_KEY",)
WORKTREE_LOCK = threading.Lock()


def worktree(tree: Path, commit: str, lock: Path) -> None:
    """Create an isolated checkout. Serialized: git takes a repo-level lock."""
    with WORKTREE_LOCK:
        for attempt in range(5):
            done = subprocess.run(
                ["git", "worktree", "add", "--detach", "-q", str(tree), commit],
                cwd=REPO, capture_output=True, text=True,
            )
            if done.returncode == 0:
                break
            time.sleep(2 + 3 * attempt)
        else:
            raise RuntimeError(f"worktree add failed 5x: {done.stderr.strip()[:200]}")
    (tree / ".venv").symlink_to(REPO / ".venv")
    (tree / ".kernelevo").mkdir(exist_ok=True)
    # Shared lock file => B300 evaluation stays serialized across worktrees.
    (tree / ".kernelevo" / "b300.lock").symlink_to(lock)


def run_arm(arm: dict, cfg: dict, logs: Path, results: list) -> None:
    tag = arm["tag"]
    url, lock_name = cfg["servers"][arm["server"]]
    tree = Path(cfg["parent"]) / f"{tag}-{''.join(random.choices('0123456789abcdef', k=8))}"
    started = time.time()
    try:
        worktree(tree, cfg["commit"], REPO / ".b300lock" / lock_name)
        cmd = [
            "./.venv/bin/python", "-u", "experiments/cute_ablation/run_iter_matrix.py",
            arm["root"], "--config", arm["config"], "--model", arm["model"],
        ]
        if arm.get("tier"):
            cmd += ["--tier", arm["tier"]]
        env = {
            **os.environ,
            "PYTHONPATH": "src",
            "CUTE_HARNESS_URL": url,
            "CUTE_AGENT_EVAL_BUDGET": str(cfg.get("eval_budget", 6)),
        }
        with (logs / f"{tag}.log").open("w") as handle:
            proc = subprocess.run(cmd, cwd=tree, env=env, stdout=handle, stderr=handle)
        ok = proc.returncode == 0
    except Exception as error:
        (logs / f"{tag}.log").write_text(f"SETUP FAILED: {error}\n")
        ok = False
    minutes = (time.time() - started) / 60
    results.append({"arm": tag, "server": arm["server"], "ok": ok,
                    "minutes": minutes, "tree": str(tree)})
    print(f"[{time.strftime('%H:%M:%S')}] {'done' if ok else 'FAILED'} {tag} "
          f"[srv {arm['server']}] ({minutes:.0f}m)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True,
                        help="JSON: {servers, caps, parent, commit, arms:[...]}")
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()

    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing env: {', '.join(missing)}")

    cfg = json.loads(args.plan.read_text(encoding="utf-8"))
    args.logs.mkdir(parents=True, exist_ok=True)
    parent = Path(cfg["parent"])
    parent.mkdir(parents=True, exist_ok=True)
    # write+execute, no read: this driver can create worktrees, an agent inside
    # one cannot list the parent and discover its siblings.
    parent.chmod(0o311)

    queues: dict[str, queue.Queue] = {}
    for arm in cfg["arms"]:
        queues.setdefault(arm["server"], queue.Queue()).put(arm)
    print(f"{len(cfg['arms'])} arms  " + "  ".join(
        f"server {s}={q.qsize()} (cap {cfg['caps'][s]})" for s, q in queues.items()), flush=True)

    results: list = []
    threads = []
    for server, work in queues.items():
        for _ in range(int(cfg["caps"][server])):
            def loop(work=work):
                while True:
                    try:
                        arm = work.get_nowait()
                    except queue.Empty:
                        return
                    run_arm(arm, cfg, args.logs, results)
            thread = threading.Thread(target=loop, daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()

    (args.logs / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"GRID COMPLETE: {sum(1 for r in results if r['ok'])}/{len(results)} ok", flush=True)


if __name__ == "__main__":
    main()
