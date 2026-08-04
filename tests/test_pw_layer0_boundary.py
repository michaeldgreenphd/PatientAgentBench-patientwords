# SPDX-License-Identifier: CC-BY-NC-4.0
"""Mechanical enforcement of the Layer 0 boundary: upstream files are never edited.

The whole integration rests on ``git pull upstream main`` staying a fast-forward.
That holds only while every PatientWords change is an *added* file: a single
edited upstream file turns every future upstream release into a merge to
resolve, and re-opens the question of what in this fork is derivative of what.

So the rule is checked here rather than trusted: take the upstream baseline
commit, list every path it contains, and assert not one of them differs in the
working tree. Adding files is allowed; changing or deleting an upstream file is
not.

The baseline is ``upstream/main`` when that remote-tracking ref exists, else the
local ``main`` (in a fresh fork clone those are the same tree). It must be
current: after ``git pull upstream main`` the fork's ``main`` advances too, and
comparing a merged branch against a stale baseline reports upstream's own
changes as violations. Outside a git checkout the test skips -- it is a
repository-shape check, not a runtime one.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Baselines in preference order.
BASELINE_REFS = ("upstream/main", "main", "origin/main")


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def resolve_baseline() -> str:
    for ref in BASELINE_REFS:
        check = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        if check.returncode == 0:
            return check.stdout.strip()
    pytest.skip(f"no upstream baseline ref among {', '.join(BASELINE_REFS)}")


@pytest.fixture(scope="module")
def baseline() -> str:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    return resolve_baseline()


@pytest.fixture(scope="module")
def upstream_paths(baseline: str) -> set:
    return set(git("ls-tree", "-r", "--name-only", baseline).splitlines())


class TestLayer0Boundary:
    def test_baseline_contains_the_upstream_package(self, upstream_paths):
        """Guards against a baseline ref that points somewhere unexpected --
        without this the emptiness assertions below would pass vacuously."""
        assert any(p.startswith("src/patient_agent_bench/") for p in upstream_paths)
        assert "pyproject.toml" in upstream_paths

    def test_no_upstream_file_is_modified(self, baseline, upstream_paths):
        changed = {
            line.split("\t", 1)[-1]
            for line in git("diff", "--name-only", baseline, "--").splitlines()
            if line.strip()
        }
        touched = sorted(changed & upstream_paths)
        assert touched == [], (
            "these upstream files differ from the baseline; Layer 0 must stay "
            f"byte-identical so upstream merges stay trivial: {touched}"
        )

    def test_no_upstream_file_is_deleted(self, upstream_paths):
        missing = sorted(p for p in upstream_paths if not (REPO_ROOT / p).exists())
        assert missing == [], f"upstream files removed from the working tree: {missing}"

    def test_upstream_package_never_depends_on_layer_1(self):
        """The dependency runs one way. Layer 1 imports upstream; nothing under
        ``src/patient_agent_bench/`` may import Layer 1, or a future upstream
        release could not be dropped in unchanged.

        Content-based rather than path-based, so it stays correct when upstream
        adds modules of its own.
        """
        offenders = []
        for path in sorted((REPO_ROOT / "src" / "patient_agent_bench").rglob("*.py")):
            if "patientwords_pab" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert offenders == [], (
            "upstream package files reference Layer 1; move the code to "
            f"src/patientwords_pab/ and register dynamically instead: {offenders}"
        )

    def test_layer_1_code_lives_in_its_own_package(self):
        """Every PatientWords module sits outside the upstream package, so
        ``git diff upstream/main -- src/patient_agent_bench`` stays empty."""
        importers = []
        for path in sorted(REPO_ROOT.rglob("*.py")):
            rel = path.relative_to(REPO_ROOT)
            if rel.parts[0] not in ("src", "tests") or ".git" in rel.parts:
                continue
            if "import patientwords_pab" not in path.read_text(encoding="utf-8"):
                continue
            in_layer_1 = rel.parts[:2] == ("src", "patientwords_pab")
            in_layer_1_tests = rel.parts[0] == "tests" and rel.name.startswith("test_pw_")
            if not (in_layer_1 or in_layer_1_tests):
                importers.append(str(rel))
        assert importers == [], (
            f"Layer 1 imported from outside its package or its tests: {importers}"
        )
