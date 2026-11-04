"""Command-line interface for trying out the face analyzer pipeline."""

import argparse
import os
from pathlib import Path

from face_analyzer.artifacts import ArtifactStore
from face_analyzer.detection import FaceDetector, crop_face
from face_analyzer.image_io import load_image, save_image
from face_analyzer.visualize import save_faces


def _resolve_model(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    """Look up the model named by --model and make sure its file is available.

    Args:
        args: Parsed arguments with manifest_path, artifact_dir and model.
        parser: Parser used to report an unknown model name.

    Returns:
        The path to the verified model file.
    """
    store = ArtifactStore.from_manifest(args.manifest_path, args.artifact_dir)
    if args.model not in store:
        parser.error(f"unknown model {args.model!r}, available: {', '.join(store.names)}")
    return store.ensure(args.model)


def _detect(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Detect faces in an image, print them and optionally save an annotated copy."""
    image = load_image(args.image)
    model_path = _resolve_model(args, parser)
    detector = FaceDetector(model_path, confidence_threshold=args.threshold)
    faces = detector.detect(image)
    if not faces:
        print("No faces found.")
    else:
        print(f"Found {len(faces)} face(s)")
    if args.output is not None:
        save_faces(image, faces, args.output)
        print(f"Saved annotated image to {args.output}")
        for i, face in enumerate(faces, start=1):
            crop = crop_face(image, face)
            save_image(crop, args.output.parent / f"{args.output.stem}_face{i}{args.output.suffix}")


def main(argv: list[str] | None = None) -> None:
    """Run the face-analyzer command-line interface.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].
    """
    parser = argparse.ArgumentParser(prog="face-analyzer")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=os.environ.get("FACE_ANALYZER_MANIFEST_PATH") or "configs/models.yaml",
        help="Path to the manifest YAML file (env: FACE_ANALYZER_MANIFEST_PATH, default: %(default)s).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=os.environ.get("FACE_ANALYZER_ARTIFACT_DIR") or "models",
        help="Directory where artifacts are stored (env: FACE_ANALYZER_ARTIFACT_DIR, default: %(default)s).",
    )
    sub = parser.add_subparsers(required=True)

    detect = sub.add_parser("detect", help="Detect faces in an image.")
    detect.add_argument("image", type=Path, help="Path to the input image.")
    detect.add_argument("--model", default="yunet", help="Model name in the manifest (default: %(default)s).")
    detect.add_argument("--threshold", type=float, default=0.9, help="Minimum detection confidence.")
    detect.add_argument(
        "-o", "--output", type=Path, help="Save a copy of the image with the detected faces drawn on it."
    )
    detect.set_defaults(func=_detect)

    args = parser.parse_args(argv)
    try:
        args.func(args, parser)
    except (OSError, ValueError) as e:
        parser.exit(1, f"{parser.prog}: error: {e}\n")
