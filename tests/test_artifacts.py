import hashlib
import urllib.error
import urllib.request

import pytest

import artifacts

CONTENT = b"model weights"
CONTENT_HASH = hashlib.sha256(CONTENT).hexdigest()


@pytest.fixture
def source(tmp_path):
    """A file served over a file:// URL, standing in for a remote artifact."""
    path = tmp_path / "remote" / "model.onnx"
    path.parent.mkdir()
    path.write_bytes(CONTENT)
    return path


def write_manifest(path, entries):
    lines = []
    for name, info in entries.items():
        lines.append(f"{name}:")
        lines.extend(f"  {key}: {value}" for key, value in info.items())
    path.write_text("\n".join(lines) + "\n")
    return path


def test_hash_matches(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(CONTENT)
    assert artifacts.hash_matches(path, CONTENT_HASH)
    assert not artifacts.hash_matches(path, "0" * 64)


def test_safe_download_writes_file(tmp_path, source):
    dest = tmp_path / "out" / "nested" / "model.onnx"
    result = artifacts.safe_download(source.as_uri(), dest, CONTENT_HASH)
    assert result == dest
    assert dest.read_bytes() == CONTENT
    assert list(dest.parent.iterdir()) == [dest]


def test_safe_download_file_is_world_readable(tmp_path, source):
    dest = tmp_path / "model.onnx"
    artifacts.safe_download(source.as_uri(), dest, CONTENT_HASH)
    assert dest.stat().st_mode & 0o777 == 0o644


def test_safe_download_passes_timeout(tmp_path, source, monkeypatch):
    real_urlopen = urllib.request.urlopen
    timeouts = []

    def urlopen(url, timeout):
        timeouts.append(timeout)
        return real_urlopen(url, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    artifacts.safe_download(source.as_uri(), tmp_path / "a", CONTENT_HASH)
    artifacts.safe_download(source.as_uri(), tmp_path / "b", CONTENT_HASH, timeout=5)
    assert timeouts == [60, 5]


def test_safe_download_hash_mismatch_leaves_nothing(tmp_path, source):
    dest = tmp_path / "out" / "model.onnx"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        artifacts.safe_download(source.as_uri(), dest, "0" * 64)
    assert list(dest.parent.iterdir()) == []


def test_safe_download_hash_mismatch_keeps_existing_file(tmp_path, source):
    dest = tmp_path / "model.onnx"
    dest.write_bytes(b"old")
    with pytest.raises(ValueError):
        artifacts.safe_download(source.as_uri(), dest, "0" * 64)
    assert dest.read_bytes() == b"old"


def test_safe_download_fetch_error_cleans_up_temp_file(tmp_path):
    dest = tmp_path / "out" / "model.onnx"
    missing = (tmp_path / "does-not-exist").as_uri()
    with pytest.raises(urllib.error.URLError):
        artifacts.safe_download(missing, dest, CONTENT_HASH)
    assert list(dest.parent.iterdir()) == []


def make_artifact(file_path, sha256=CONTENT_HASH):
    return artifacts.Artifact(name="yunet", url="https://example.com/m.onnx", sha256=sha256, file_path=str(file_path))


def test_artifact_exists(tmp_path):
    path = tmp_path / "f"
    assert not artifacts.artifact_exists(make_artifact(path))
    path.write_bytes(CONTENT)
    assert artifacts.artifact_exists(make_artifact(path))
    assert not artifacts.artifact_exists(make_artifact(path, sha256="0" * 64))


def test_artifact_exists_directory(tmp_path):
    assert not artifacts.artifact_exists(make_artifact(tmp_path))


def test_load_manifest(tmp_path):
    manifest = write_manifest(
        tmp_path / "m.yaml",
        {"yunet": {"url": "https://example.com/m.onnx", "sha256": CONTENT_HASH, "filename": "m.onnx"}},
    )
    assert artifacts.load_manifest(manifest, tmp_path / "models") == {
        "yunet": artifacts.Artifact(
            name="yunet",
            url="https://example.com/m.onnx",
            sha256=CONTENT_HASH,
            file_path=str(tmp_path / "models" / "m.onnx"),
        )
    }


@pytest.mark.parametrize("text", ["", "{}\n", "- a\n- b\n", "just a string\n"])
def test_load_manifest_rejects_non_mapping(tmp_path, text):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(text)
    with pytest.raises(ValueError, match="must be a non-empty mapping"):
        artifacts.load_manifest(manifest, tmp_path)


@pytest.mark.parametrize("text", ["yunet:\n", "yunet: https://example.com/m.onnx\n", "yunet: [a, b]\n"])
def test_load_manifest_rejects_non_mapping_entry(tmp_path, text):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(text)
    with pytest.raises(ValueError, match="entry for yunet must be a mapping"):
        artifacts.load_manifest(manifest, tmp_path)


@pytest.mark.parametrize("missing", ["url", "sha256", "filename"])
def test_load_manifest_missing_key(tmp_path, missing):
    info = {"url": "https://example.com/m.onnx", "sha256": CONTENT_HASH, "filename": "m.onnx"}
    del info[missing]
    manifest = write_manifest(tmp_path / "m.yaml", {"yunet": info})
    with pytest.raises(ValueError, match="missing required keys"):
        artifacts.load_manifest(manifest, tmp_path)


@pytest.mark.parametrize("sha256", ["abc", "g" * 64, "a" * 63, "a" * 65])
def test_load_manifest_invalid_sha256(tmp_path, sha256):
    manifest = write_manifest(
        tmp_path / "m.yaml", {"yunet": {"url": "https://example.com/m.onnx", "sha256": sha256, "filename": "m.onnx"}}
    )
    with pytest.raises(ValueError, match="invalid sha256 format"):
        artifacts.load_manifest(manifest, tmp_path)


def test_load_manifest_normalizes_sha256(tmp_path):
    manifest = write_manifest(
        tmp_path / "m.yaml",
        {"yunet": {"url": "https://example.com/m.onnx", "sha256": CONTENT_HASH.upper(), "filename": "m.onnx"}},
    )
    assert artifacts.load_manifest(manifest, tmp_path)["yunet"].sha256 == CONTENT_HASH


def test_process_manifest_downloads_missing(tmp_path, source, capsys):
    manifest = write_manifest(
        tmp_path / "m.yaml", {"yunet": {"url": source.as_uri(), "sha256": CONTENT_HASH, "filename": "model.onnx"}}
    )
    artifacts.process_manifest(manifest, tmp_path / "models")
    assert (tmp_path / "models" / "model.onnx").read_bytes() == CONTENT
    assert "Downloading yunet" in capsys.readouterr().out


def test_process_manifest_skips_valid_file(tmp_path, monkeypatch, capsys):
    (tmp_path / "model.onnx").write_bytes(CONTENT)
    manifest = write_manifest(
        tmp_path / "m.yaml",
        {"yunet": {"url": "https://invalid.example", "sha256": CONTENT_HASH, "filename": "model.onnx"}},
    )
    monkeypatch.setattr(artifacts, "safe_download", lambda *a: pytest.fail("should not download"))
    artifacts.process_manifest(manifest, tmp_path)
    assert "already exists" in capsys.readouterr().out


def test_process_manifest_replaces_outdated_file(tmp_path, source):
    dest = tmp_path / "model.onnx"
    dest.write_bytes(b"stale")
    manifest = write_manifest(
        tmp_path / "m.yaml", {"yunet": {"url": source.as_uri(), "sha256": CONTENT_HASH, "filename": "model.onnx"}}
    )
    artifacts.process_manifest(manifest, tmp_path)
    assert dest.read_bytes() == CONTENT


def test_main_reads_cli_args(tmp_path, source, monkeypatch):
    manifest = write_manifest(
        tmp_path / "m.yaml", {"yunet": {"url": source.as_uri(), "sha256": CONTENT_HASH, "filename": "model.onnx"}}
    )
    artifact_dir = tmp_path / "models"
    monkeypatch.setattr(
        "sys.argv", ["artifacts", "--manifest-path", str(manifest), "--artifact-dir", str(artifact_dir)]
    )
    artifacts.main()
    assert (artifact_dir / "model.onnx").read_bytes() == CONTENT


def test_main_reads_env_vars(tmp_path, source, monkeypatch):
    manifest = write_manifest(
        tmp_path / "m.yaml", {"yunet": {"url": source.as_uri(), "sha256": CONTENT_HASH, "filename": "model.onnx"}}
    )
    artifact_dir = tmp_path / "models"
    monkeypatch.setenv("MANIFEST_PATH", str(manifest))
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setattr("sys.argv", ["artifacts"])
    artifacts.main()
    assert (artifact_dir / "model.onnx").read_bytes() == CONTENT


def test_main_cli_args_override_env_vars(tmp_path, source, monkeypatch):
    manifest = write_manifest(
        tmp_path / "m.yaml", {"yunet": {"url": source.as_uri(), "sha256": CONTENT_HASH, "filename": "model.onnx"}}
    )
    artifact_dir = tmp_path / "models"
    monkeypatch.setenv("MANIFEST_PATH", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "unused"))
    monkeypatch.setattr(
        "sys.argv", ["artifacts", "--manifest-path", str(manifest), "--artifact-dir", str(artifact_dir)]
    )
    artifacts.main()
    assert (artifact_dir / "model.onnx").read_bytes() == CONTENT
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    ("env", "missing"),
    [
        ({}, "--manifest-path, --artifact-dir"),
        ({"MANIFEST_PATH": "", "ARTIFACT_DIR": ""}, "--manifest-path, --artifact-dir"),
        ({"MANIFEST_PATH": "m.yaml"}, "--artifact-dir"),
        ({"ARTIFACT_DIR": "models"}, "--manifest-path"),
    ],
)
def test_main_requires_args_without_env_vars(monkeypatch, capsys, env, missing):
    monkeypatch.delenv("MANIFEST_PATH", raising=False)
    monkeypatch.delenv("ARTIFACT_DIR", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("sys.argv", ["artifacts"])
    with pytest.raises(SystemExit) as exc_info:
        artifacts.main()
    assert exc_info.value.code == 2
    assert f"the following arguments are required: {missing}" in capsys.readouterr().err
