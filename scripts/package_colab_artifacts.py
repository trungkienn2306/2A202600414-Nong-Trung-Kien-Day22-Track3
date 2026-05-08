#!/usr/bin/env python3
"""Package Colab outputs for download back to the local repo."""
from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


@dataclass(frozen=True)
class ArtifactCheck:
    path: str
    required: bool = True


class ArtifactStatus(TypedDict):
    path: str
    required: bool
    exists: bool
    size_mb: float


class GgufStatus(TypedDict):
    path: str
    size_mb: float


class Manifest(TypedDict):
    repo_root: str
    checks: list[ArtifactStatus]
    gguf_files: list[GgufStatus]
    missing_required: list[str]
    required_artifacts_present: bool


ARTIFACTS = (
    ArtifactCheck("adapters/sft-mini/adapter_config.json"),
    ArtifactCheck("adapters/dpo/adapter_config.json"),
    ArtifactCheck("adapters/dpo/dpo_metrics.json"),
    ArtifactCheck("data/pref/train.parquet"),
    ArtifactCheck("data/eval/side_by_side.jsonl"),
    ArtifactCheck("data/eval/judge_results.json"),
    ArtifactCheck("data/eval/benchmark_results.json"),
    ArtifactCheck("submission/screenshots/07-benchmark-comparison.png"),
    ArtifactCheck("submission/REFLECTION.md", required=False),
)

ADAPTER_SUFFIXES = {".json", ".safetensors", ".bin", ".model", ".spm", ".jinja"}
DATA_SUFFIXES = {".json", ".jsonl", ".parquet"}
NOTEBOOK_SUFFIXES = {".ipynb", ".py"}
SCREENSHOT_SUFFIXES = {".png", ".jpg", ".jpeg"}

EXCLUDED_SUFFIXES = {".bak", ".tmp", ".key", ".pem", ".p12", ".pfx"}
EXCLUDED_NAMES = {
    ".env",
    "day22-colab-artifacts.zip",
    "credentials.json",
    "kaggle.json",
    "service-account.json",
    "token.json",
}


def resolve_repo_root(value: str | None) -> Path:
    root = (Path(value).expanduser() if value else Path.cwd()).resolve()
    if not root.is_dir():
        raise ValueError(f"--repo-root must be an existing directory: {root}")
    return root


def validate_archive_name(value: str) -> str:
    archive = Path(value)
    if archive.is_absolute() or archive.name != value or ".." in archive.parts:
        raise ValueError("--output must be a simple filename, not a path")
    if archive.suffix.lower() != ".zip":
        raise ValueError("--output must end with .zip")
    return value


def existing_file_status(root: Path, relative_path: str) -> tuple[bool, float]:
    path = root / relative_path
    if not path.is_file():
        return False, 0.0
    return path.stat().st_size > 0, round(path.stat().st_size / 1_000_000, 3)


def collect_status(root: Path) -> Manifest:
    checks: list[ArtifactStatus] = []
    missing_required: list[str] = []
    for artifact in ARTIFACTS:
        exists, size_mb = existing_file_status(root, artifact.path)
        checks.append(
            {
                "path": artifact.path,
                "required": artifact.required,
                "exists": exists,
                "size_mb": size_mb,
            }
        )
        if artifact.required and not exists:
            missing_required.append(artifact.path)

    gguf_files = sorted((root / "gguf").glob("*.gguf")) if (root / "gguf").exists() else []
    q4_gguf_files = [path for path in gguf_files if path.is_file() and "q4_k_m" in path.name.lower()]
    if not q4_gguf_files:
        missing_required.append("gguf/*Q4_K_M*.gguf")

    return {
        "repo_root": str(root),
        "checks": checks,
        "gguf_files": [
            {"path": str(path.relative_to(root)), "size_mb": round(path.stat().st_size / 1_000_000, 1)}
            for path in gguf_files
            if path.is_file()
        ],
        "missing_required": missing_required,
        "required_artifacts_present": not missing_required,
    }


def is_hidden_or_secret(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    if any(part.startswith(".") for part in lowered_parts):
        return True
    if path.name.lower() in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return any(part.startswith(".env") for part in lowered_parts)


def matches_allowlist(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    parts = relative.parts
    suffix = path.suffix.lower()

    if len(parts) >= 2 and parts[0] == "adapters" and parts[1] in {"sft-mini", "dpo"}:
        return suffix in ADAPTER_SUFFIXES
    if len(parts) >= 2 and parts[0] == "data" and parts[1] in {"pref", "eval"}:
        return suffix in DATA_SUFFIXES
    if len(parts) == 2 and parts[0] == "gguf":
        return suffix == ".gguf"
    if len(parts) == 2 and parts[0] == "notebooks":
        return suffix in NOTEBOOK_SUFFIXES
    if parts == ("submission", "REFLECTION.md"):
        return True
    if len(parts) == 3 and parts[0] == "submission" and parts[1] == "screenshots":
        return suffix in SCREENSHOT_SUFFIXES
    if parts == ("submission", "colab_artifact_manifest.json"):
        return True
    return False


def is_safe_file(root: Path, path: Path) -> bool:
    if path.is_symlink() or not path.is_file() or is_hidden_or_secret(path):
        return False
    try:
        path.resolve().relative_to(root)
        path.relative_to(root)
    except ValueError:
        return False
    return matches_allowlist(root, path)


def iter_package_files(root: Path) -> list[Path]:
    candidates = (
        list((root / "adapters").rglob("*"))
        + list((root / "data").rglob("*"))
        + list((root / "gguf").glob("*"))
        + list((root / "notebooks").glob("*"))
        + list((root / "submission").rglob("*"))
    )
    return sorted(path for path in candidates if is_safe_file(root, path))


def write_manifest(root: Path, manifest: Manifest) -> Path:
    output = root / "submission" / "colab_artifact_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def create_archive(root: Path, archive_name: str) -> Path:
    archive_name = validate_archive_name(archive_name)
    archive_path = root / archive_name
    temp_path = root / f".{archive_name}.tmp"
    files = iter_package_files(root)
    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
    temp_path.replace(archive_path)
    return archive_path


def download_in_colab(archive_path: Path) -> bool:
    try:
        from google.colab import files  # type: ignore[import-not-found]
    except ImportError:
        print("\n--download was requested, but this is not a Google Colab runtime.")
        return False
    files.download(str(archive_path))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Package Day 22 Colab artifacts into a zip file.")
    parser.add_argument("--repo-root", help="Repo root in Colab, e.g. /content/lab22")
    parser.add_argument("--output", default="day22-colab-artifacts.zip", help="Archive filename")
    parser.add_argument("--download", action="store_true", help="Trigger browser download in Google Colab")
    args = parser.parse_args()

    root = resolve_repo_root(args.repo_root)
    manifest = collect_status(root)
    manifest_path = write_manifest(root, manifest)
    archive_path = create_archive(root, args.output)

    print(f"Manifest: {manifest_path}")
    print(f"Archive:  {archive_path}")
    print(f"Size:     {archive_path.stat().st_size / 1_000_000:.1f} MB")

    if manifest["missing_required"]:
        print("\nMissing required artifacts:")
        for path in manifest["missing_required"]:
            print(f"  - {path}")
        print("\nPackage was still created so you can inspect partial Colab outputs.")
        return 1

    print("\nAll required artifact checks passed. Inspect the manifest before downloading or submitting.")
    if args.download:
        download_in_colab(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
