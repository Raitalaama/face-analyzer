import argparse
import contextlib
import hashlib
import os
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    sha256: str
    file_path: Path


def file_sha256(path: str | Path) -> str:
    """Return the sha256 hash of the file at path."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def hash_matches(path: str | Path, expected_hash: str) -> bool:
    """Check if the file at path has the expected sha256 hash."""
    return file_sha256(path) == expected_hash


def safe_download(url: str, dest: str | Path, expected_hash: str, timeout: float = 60) -> Path:
    """Download url to dest, verifying its sha256.

    The file is written to a temp file in dest's directory and only moved
    into place once the hash matches, so dest never holds a partial or
    corrupt download. timeout (seconds) applies to connecting and to each
    read, not to the whole download.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url, timeout=timeout) as resp:
            shutil.copyfileobj(resp, f)
        if not hash_matches(tmp, expected_hash):
            raise ValueError(f"sha256 mismatch for {url}, expected {expected_hash}, got {file_sha256(tmp)}")
        # mkstemp creates the file as 0600; make it readable like a normally created file.
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return dest


def artifact_exists(artifact: Artifact) -> bool:
    """Check if a file exists at the given path and has the expected hash."""
    return artifact.file_path.is_file() and hash_matches(artifact.file_path, artifact.sha256)


def load_manifest(manifest_path: str | Path, artifact_dir: str | Path) -> dict[str, Artifact]:
    """Load a manifest file and return its contents as a dictionary."""
    artifacts: dict[str, Artifact] = {}
    with open(manifest_path) as f:
        manifest_dict = yaml.safe_load(f)
    if not isinstance(manifest_dict, dict) or not manifest_dict:
        raise ValueError(f"Manifest {manifest_path} must be a non-empty mapping of artifact names to entries.")
    for name, info in manifest_dict.items():
        if not isinstance(info, dict):
            raise ValueError(f"Manifest entry for {name} must be a mapping.")
        if not all(k in info for k in ("url", "sha256", "filename")):
            raise ValueError(f"Manifest entry for {name} is missing required keys.")
        sha256 = str(info["sha256"]).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"Manifest entry for {name} has an invalid sha256 format.")
        artifacts[name] = Artifact(
            name=name,
            url=info["url"],
            sha256=sha256,
            file_path=Path(artifact_dir) / info["filename"],
        )
    return artifacts


def process_manifest(manifest_path: str | Path, artifact_dir: str | Path) -> None:
    """Process a manifest file, downloading any missing or outdated files."""
    manifest = load_manifest(manifest_path, artifact_dir)
    for name, artifact in manifest.items():
        if not artifact_exists(artifact):
            print(f"Downloading {name} from {artifact.url} to {artifact.file_path}...")
            safe_download(artifact.url, artifact.file_path, artifact.sha256)
        else:
            print(f"{name} already exists and matches the expected hash.")


def main() -> None:
    """Main function to process the manifest file."""
    parser = argparse.ArgumentParser()
    manifest_env = os.environ.get("FACE_ANALYZER_MANIFEST_PATH") or None
    artifact_dir_env = os.environ.get("FACE_ANALYZER_ARTIFACT_DIR") or None
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=manifest_env,
        required=manifest_env is None,
        help="Path to the manifest YAML file (env: FACE_ANALYZER_MANIFEST_PATH).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=artifact_dir_env,
        required=artifact_dir_env is None,
        help="Directory to store downloaded artifacts (env: FACE_ANALYZER_ARTIFACT_DIR).",
    )
    args = parser.parse_args()
    process_manifest(args.manifest_path, args.artifact_dir)


if __name__ == "__main__":
    main()
