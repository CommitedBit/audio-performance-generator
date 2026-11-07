"""scripts/check.sh --full: its dev run must never delete anyone else's results.

smoke-results/ is gitignored and may hold the owner's GPU runs, the only
record of them. check.sh once ended with `rm -rf smoke-results`. A later
version removed every directory that appeared while it ran, which included a
GPU run started meanwhile. Now the dev run writes to a directory of its own
(`smoke_gpu.py --out`), and check.sh removes exactly that one, and only when
the run passes.

These tests run the real script under bash, with every tool it calls replaced
by a stub on PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# python3 stands in for pytest, ruff, check_deps.py and snapshot_contracts.py,
# which all pass, and for smoke_gpu.py. As that, it writes the run directory
# (--out, or a timestamped one like the real script's default) and, with
# GPU_RUN set, starts the owner's GPU run in smoke-results/ while it runs.
PYTHON3 = r"""#!/bin/sh
case " $* " in
  *" scripts/smoke_gpu.py "*) ;;
  *) exit 0 ;;
esac
out="smoke-results/$(date +%Y%m%d-%H%M%S)"
prev=
for a in "$@"; do
  [ "$prev" = --out ] && out=$a
  prev=$a
done
echo "$out" > "$OUT_LOG"
if [ -n "${GPU_RUN:-}" ]; then
  mkdir -p "smoke-results/$GPU_RUN" && echo gpu > "smoke-results/$GPU_RUN/report.txt"
fi
mkdir -p "$(dirname "$out")" && mkdir "$out" && echo dev > "$out/report.txt" || exit 4
exit "${SMOKE_EXIT:-0}"
"""
PASS = "#!/bin/sh\nexit 0\n"
EARLIER_GPU_RUN = "20200101-000000"


@pytest.fixture
def repo(tmp_path):
    """check.sh in an otherwise empty checkout, with stub tools on PATH."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "backend").mkdir()
    shutil.copy2(ROOT / "scripts" / "check.sh", repo / "scripts" / "check.sh")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in {"python3": PYTHON3, "npm": PASS, "docker": PASS, "uv": PASS}.items():
        (stubs / name).write_text(body)
        (stubs / name).chmod(0o755)
    return repo


def run_check(repo: Path, **env: str) -> tuple[subprocess.CompletedProcess, Path]:
    out_log = repo.parent / "dev-run.txt"
    result = subprocess.run(
        ["bash", "scripts/check.sh", "--full"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
        # /bin/bash is bash 3.2 on macOS, the version check.sh is written for.
        env={"PATH": f"{repo.parent / 'bin'}:/usr/bin:/bin", "HOME": str(repo.parent),
             "OUT_LOG": str(out_log), **env},
    )
    assert out_log.exists(), result.stdout + result.stderr
    return result, repo / out_log.read_text().strip()


def earlier_gpu_run(repo: Path) -> Path:
    run = repo / "smoke-results" / EARLIER_GPU_RUN
    run.mkdir(parents=True)
    (run / "report.txt").write_text("gpu\n")
    return run


def test_results_already_there_survive_a_passing_dev_run(repo):
    kept = earlier_gpu_run(repo)

    result, dev_run = run_check(repo)

    assert result.returncode == 0, result.stdout
    assert (kept / "report.txt").read_text() == "gpu\n"
    assert not dev_run.exists()
    assert os.listdir(repo / "smoke-results") == [EARLIER_GPU_RUN]


def test_a_gpu_run_started_during_the_dev_run_survives(repo):
    result, dev_run = run_check(repo, GPU_RUN="20991231-235959")

    assert result.returncode == 0, result.stdout
    assert (repo / "smoke-results" / "20991231-235959" / "report.txt").read_text() == "gpu\n"
    assert not dev_run.exists()


def test_a_failed_dev_run_keeps_its_results_and_says_where(repo):
    kept = earlier_gpu_run(repo)

    result, dev_run = run_check(repo, SMOKE_EXIT="1")

    assert result.returncode == 1
    assert (dev_run / "report.txt").read_text() == "dev\n"
    assert dev_run.name in result.stdout
    assert (kept / "report.txt").read_text() == "gpu\n"


def test_a_passing_dev_run_leaves_no_smoke_results_behind(repo):
    result, dev_run = run_check(repo)

    assert result.returncode == 0, result.stdout
    assert not dev_run.exists()
    assert not (repo / "smoke-results").exists()
