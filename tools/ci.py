"""Install a checkout in isolation, test its built artifact and prepare a release.

Only hash-locked release dependencies are downloaded. This script never reads
sibling repositories, datasets, credentials files, model weights or GPUs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".ci"
CONFIG = json.loads((ROOT / "ci.json").read_text())


def run(*command: str, cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def python_at(environment: Path) -> str:
    return str(environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


def private_dependencies() -> Path:
    cache = CACHE / "wheels"
    cache.mkdir(parents=True, exist_ok=True)
    for entry in CONFIG["dependencies"]:
        target = cache / entry["asset"]
        if not target.exists():
            run("gh", "release", "download", entry["release"], "--repo", entry["repository"],
                "--pattern", entry["asset"], "--dir", str(cache))
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit("Release dependency hash mismatch: " + entry["asset"])
    return cache


def audit_tracked_files() -> None:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in filter(None, names):
        lower = name.lower()
        if (Path(name).name == ".env" or re.search(r"\.(safetensors|pt|pth|ckpt)$", lower)
                or any(part in ("datasets", "checkpoints", "weights") for part in Path(lower).parts)
                or (ROOT / name).stat().st_size > 32 * 1024**2):
            raise SystemExit("Unadmitted Git content: " + name)


def npm(*args: str, cwd: Path = ROOT) -> None:
    run("npm.cmd" if os.name == "nt" else "npm", *args, cwd=cwd)


def verify() -> None:
    CACHE.mkdir(exist_ok=True)
    audit_tracked_files()
    dependencies = private_dependencies()
    reports = ROOT / "ci-results"
    reports.mkdir(exist_ok=True)
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    kind = CONFIG["kind"]
    if kind == "python":
        build_env, test_env = CACHE / "build", CACHE / "test"
        run("uv", "venv", str(build_env), "--python", sys.executable)
        run("uv", "pip", "sync", "--python", python_at(build_env), "--require-hashes",
            "requirements.build.lock.txt")
        run("uv", "build", "--wheel", "--sdist", "--no-build-isolation", "--python", python_at(build_env),
            "--out-dir", str(dist))
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        # Old locks contain the previously released component. Remove ONLY self;
        # the newly built wheel must be the component exercised by this checkout.
        source = (ROOT / CONFIG["requirements"]).read_text()
        blocks = re.split(r"(?=^[A-Za-z0-9])", source, flags=re.M)
        own_name = re.sub(r"[-_.]+", "-", project["name"]).lower()
        filtered = "".join(block for block in blocks if not (
            re.match(r"[A-Za-z0-9_.-]+==", block) and
            re.sub(r"[-_.]+", "-", block.split("==", 1)[0]).lower() == own_name))
        requirements = CACHE / "requirements.tests.txt"
        requirements.write_text(filtered)
        run("uv", "venv", str(test_env), "--python", sys.executable)
        run("uv", "pip", "sync", "--python", python_at(test_env), "--require-hashes",
            "--find-links", str(dependencies),
            "--find-links", "https://download.pytorch.org/whl/cpu/torch/",
            "--find-links", "https://download.pytorch.org/whl/cpu/torchvision/", str(requirements))
        wheels = list(dist.glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit("Expected one freshly built component wheel")
        run("uv", "pip", "install", "--python", python_at(test_env), "--no-deps", str(wheels[0]))
        # -I excludes CWD/PYTHONPATH; overriding pytest's legacy pythonpath avoids
        # testing src/ while advertising a successful wheel installation.
        run(python_at(test_env), "-I", "-m", "pytest", "-o", "pythonpath=",
            "--junitxml=" + str(reports / "pytest.xml"), "-q", *CONFIG["tests"])
        if CONFIG.get("migrations"):
            run(python_at(test_env), "-I", "-m", "ruff", "check", "src", "tests", "migrations")
            run(python_at(test_env), "-I", "-m", "mypy")
            run(python_at(test_env), "-I", "-m", "alembic", "upgrade", "head", "--sql")
        if CONFIG.get("typescript_package"):
            npm("pack", "--pack-destination", str(dist))
        if CONFIG.get("producer_ui"):
            npm("ci", "--ignore-scripts")
            npm("ci", "--ignore-scripts", cwd=ROOT / "packages/ui")
            npm("ci", "--ignore-scripts", cwd=ROOT / "apps/web")
            npm("test", cwd=ROOT / "packages/ui")
            npm("run", "build", cwd=ROOT / "apps/web")
            npm("pack", "--pack-destination", str(dist), cwd=ROOT / "packages/ui")
        version = project["version"]
    elif kind == "frontend":
        npm("ci")
        npm("run", "check")
        npm("test")
        npm("run", "build")
        run("npx.cmd" if os.name == "nt" else "npx", "playwright", "install", "--with-deps", "chromium")
        npm("run", "test:e2e")
        shutil.make_archive(str(ROOT / "frontend-build"), "zip", dist)
        shutil.move(ROOT / "frontend-build.zip", dist / "frontend-build.zip")
        version = json.loads((ROOT / "package.json").read_text())["version"]
    elif kind == "unreal-source":
        environment = CACHE / "test"
        run("uv", "venv", str(environment), "--python", sys.executable)
        run("uv", "pip", "sync", "--python", python_at(environment), "--require-hashes", CONFIG["requirements"])
        run(python_at(environment), "tools/install_map_builder_plugin.py", str(dependencies / CONFIG["plugin_asset"]))
        run(python_at(environment), "-I", "-m", "compileall", "-q", "tools", "Source", "Plugins")
        run(python_at(environment), "-I", "-m", "pytest", "--junitxml=" + str(reports / "pytest.xml"), "tests", "-q")
        version = json.loads((ROOT / "component.json").read_text())["version"]
        run("git", "archive", "--format=zip", "--output=" + str(dist / "unreal-source.zip"), "HEAD")
    else:
        raise SystemExit("Unknown verification kind")
    artifacts = [{"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "bytes": path.stat().st_size} for path in sorted(dist.iterdir()) if path.is_file()]
    manifest = {"version": version, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
                "kind": kind, "artifacts": artifacts, "validation": "installed-artifact-tests-passed",
                "limits": CONFIG.get("limits", [])}
    (dist / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dist / "RELEASE.md").write_text(f"Version {version}. Installation isolée et contrôles automatisés réussis.\n\n" +
                                    "\n".join(CONFIG.get("limits", [])) + "\n", encoding="utf-8")


def release() -> None:
    manifest = json.loads((ROOT / "dist/release-manifest.json").read_text())
    tag = os.environ["GITHUB_REF_NAME"]
    if tag != "v" + manifest["version"] or os.environ["GITHUB_SHA"] != manifest["commit"]:
        raise SystemExit("Tag, source commit and tested version differ")
    run("git", "fetch", "origin", "main", "--no-tags")
    run("git", "merge-base", "--is-ancestor", manifest["commit"], "origin/main")
    for item in manifest["artifacts"]:
        path = ROOT / "dist" / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit("Release artifact hash mismatch")
    # No --clobber: an accepted version is never replaced by a repeated run.
    run("gh", "release", "create", tag, "--verify-tag", "--title", tag,
        "--notes-file", str(ROOT / "dist/RELEASE.md"),
        *[str(p) for p in (ROOT / "dist").iterdir() if p.is_file()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "release"])
    arguments = parser.parse_args()
    (verify if arguments.command == "verify" else release)()
