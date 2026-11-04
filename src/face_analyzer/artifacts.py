"""Download and verify model artifacts listed in a YAML manifest."""

import argparse
import contextlib
import hashlib
import os
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml


@dataclass(frozen=True)
class Artifact:
    """A downloadable file described by a manifest entry.

    Attributes:
        name: Artifact name, the key of its manifest entry.
        url: URL the file is downloaded from.
        sha256: Expected lowercase hex sha256 digest of the file.
        file_path: Local path where the file is stored.
    """

    name: str
    url: str
    sha256: str
    file_path: Path


def file_sha256(path: str | Path) -> str:
    """Compute the sha256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        The lowercase hex sha256 digest of the file contents.
    """
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def hash_matches(path: str | Path, expected_hash: str) -> bool:
    """Check whether a file has the expected sha256 hash.

    Args:
        path: Path to the file.
        expected_hash: Expected lowercase hex sha256 digest.

    Returns:
        True if the file's digest equals expected_hash.
    """
    return file_sha256(path) == expected_hash


def safe_download(url: str, dest: str | Path, expected_hash: str, timeout: float = 60) -> Path:
    """Download a file and verify its sha256 before moving it into place.

    The file is written to a temp file in dest's directory and only moved
    into place once the hash matches, so dest never holds a partial or
    corrupt download.

    Args:
        url: URL to download from.
        dest: Destination file path. Parent directories are created if missing.
        expected_hash: Expected lowercase hex sha256 digest of the file.
        timeout: Seconds allowed for connecting and for each read, not for
            the whole download.

    Returns:
        The destination path.

    Raises:
        ValueError: If the downloaded file's hash does not match expected_hash.
        urllib.error.URLError: If the download fails.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url, timeout=timeout) as resp:
            shutil.copyfileobj(resp, f)
        actual_hash = file_sha256(tmp)
        if actual_hash != expected_hash:
            raise ValueError(f"sha256 mismatch for {url}, expected {expected_hash}, got {actual_hash}")
        # mkstemp creates the file as 0600; make it readable like a normally created file.
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return dest


def artifact_exists(artifact: Artifact) -> bool:
    """Check whether an artifact's file exists and has the expected hash.

    Args:
        artifact: The artifact to check.

    Returns:
        True if artifact.file_path is a file whose sha256 matches artifact.sha256.
    """
    return artifact.file_path.is_file() and hash_matches(artifact.file_path, artifact.sha256)


def load_manifest(manifest_path: str | Path, artifact_dir: str | Path) -> dict[str, Artifact]:
    """Load and validate a manifest file.

    The manifest is a YAML mapping of artifact names to entries, each with
    ``url``, ``sha256`` and ``filename`` keys.

    Args:
        manifest_path: Path to the manifest YAML file.
        artifact_dir: Directory the artifact filenames are resolved against.

    Returns:
        A mapping of artifact names to Artifact objects.

    Raises:
        ValueError: If the manifest is not a non-empty mapping, or an entry is
            not a mapping, is missing required keys, or has an invalid sha256.
    """
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


def ensure_artifact(artifact: Artifact) -> Path:
    """Ensure that an artifact's file exists and has the expected hash.

    If the file does not exist or its hash does not match, it is downloaded
    from the artifact's URL.

    Args:
        artifact: The artifact to ensure.

    Returns:
        The path to the verified file.

    Raises:
        ValueError: If the downloaded file's hash does not match.
        urllib.error.URLError: If the download fails.
    """
    if artifact_exists(artifact):
        print(f"{artifact.name} already exists and matches the expected hash.")
    else:
        print(f"Downloading {artifact.name} from {artifact.url} to {artifact.file_path}...")
        safe_download(artifact.url, artifact.file_path, artifact.sha256)
    return artifact.file_path


class ArtifactStore:
    """The artifacts of one manifest, looked up and fetched by name.

    Example:
        store = ArtifactStore.from_manifest("configs/models.yaml", "models")
        model_path = store.ensure("yunet")
    """

    def __init__(self, artifacts: Mapping[str, Artifact]) -> None:
        """Create a store from already loaded artifacts.

        Args:
            artifacts: Mapping of artifact names to Artifact objects.
        """
        self._artifacts = dict(artifacts)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, artifact_dir: str | Path) -> Self:
        """Create a store from a manifest file.

        Args:
            manifest_path: Path to the manifest YAML file.
            artifact_dir: Directory where artifacts are stored.

        Returns:
            A store holding the manifest's artifacts.

        Raises:
            ValueError: If the manifest is invalid.
        """
        return cls(load_manifest(manifest_path, artifact_dir))

    @property
    def names(self) -> list[str]:
        """Names of the artifacts in the store, in manifest order."""
        return list(self._artifacts)

    def __contains__(self, name: object) -> bool:
        """Check whether an artifact with the given name exists in the store."""
        return name in self._artifacts

    def __iter__(self) -> Iterator[Artifact]:
        """Iterate over the artifacts in the store, in manifest order."""
        return iter(self._artifacts.values())

    def __len__(self) -> int:
        """Return the number of artifacts in the store."""
        return len(self._artifacts)

    def get(self, name: str) -> Artifact:
        """Look up an artifact by name.

        Args:
            name: Artifact name, the key of its manifest entry.

        Returns:
            The artifact.

        Raises:
            KeyError: If no artifact has that name. The message lists the
                available names.
        """
        try:
            return self._artifacts[name]
        except KeyError:
            available = ", ".join(self.names)
            raise KeyError(f"Unknown artifact {name!r}. Available: {available}") from None

    def ensure(self, name: str) -> Path:
        """Ensure that an artifact's file exists and is valid, downloading it if needed.

        Args:
            name: Artifact name, the key of its manifest entry.

        Returns:
            The path to the verified file.

        Raises:
            KeyError: If no artifact has that name.
            ValueError: If the downloaded file's hash does not match.
            urllib.error.URLError: If the download fails.
        """
        return ensure_artifact(self.get(name))

    def ensure_all(self) -> dict[str, Path]:
        """Ensure every artifact in the store, downloading any that are missing or outdated.

        Returns:
            A mapping of artifact names to the paths of their verified files.
        """
        return {artifact.name: ensure_artifact(artifact) for artifact in self}


def process_manifest(manifest_path: str | Path, artifact_dir: str | Path) -> None:
    """Download any artifacts in a manifest that are missing or outdated.

    Args:
        manifest_path: Path to the manifest YAML file.
        artifact_dir: Directory where artifacts are stored.

    Raises:
        ValueError: If the manifest is invalid or a downloaded file's hash
            does not match.
    """
    ArtifactStore.from_manifest(manifest_path, artifact_dir).ensure_all()


def main() -> None:
    """Download the artifacts listed in a manifest file.

    The manifest path and artifact directory are read from the
    ``--manifest-path`` and ``--artifact-dir`` command-line options, falling
    back to the FACE_ANALYZER_MANIFEST_PATH and FACE_ANALYZER_ARTIFACT_DIR
    environment variables.
    """
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
