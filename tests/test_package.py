from importlib.metadata import version

import face_analyzer


def test_version_is_exposed():
    assert isinstance(face_analyzer.__version__, str)


def test_version_matches_package_metadata():
    assert face_analyzer.__version__ == version("face-analyzer")


def test_version_format():
    """Releases use plain MAJOR.MINOR.PATCH, no pre-release suffixes."""
    parts = face_analyzer.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdecimal() for part in parts)
