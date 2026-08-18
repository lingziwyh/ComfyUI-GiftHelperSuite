from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path


LOGGER = logging.getLogger(__name__)

EXAMPLE_ASSET_DIR = Path(__file__).resolve().parent / "example_workflows" / "assets"

# Public filenames are intentionally package-prefixed because ComfyUI and
# VideoHelperSuite expose files from the root of ComfyUI/input in their loader
# dropdowns. Unique names keep the examples portable without colliding with a
# user's own BG.png or AnimateDiff exports.
EXAMPLE_ASSET_MANIFEST = {
    "GiftHelperSuite_PostFX_Source_bebfe94a.mp4": {
        "size": 12_360_569,
        "sha256": "bebfe94a156c096b70d0964ef3ca14df6aff61b7a0ea57aaf5b83622fde988a8",
    },
    "GiftHelperSuite_Chroma_Source_92532474.mp4": {
        "size": 1_013_054,
        "sha256": "925324744d8e513337c1fe5e3a190281d7e75b605ad8b48a164d4e14571d6535",
    },
    "GiftHelperSuite_Example_Background_213ee241.png": {
        "size": 1_273_999,
        "sha256": "213ee241e04fb28c1050ec9fcdd901d239cc354750231028b9e559bc28f3eb89",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_manifest(path: Path, metadata: dict[str, int | str]) -> bool:
    try:
        if path.stat().st_size != metadata["size"]:
            return False
        return _sha256(path) == metadata["sha256"]
    except OSError:
        return False


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def install_example_assets(input_dir: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Install bundled demo media into ComfyUI/input without overwriting files.

    The operation is idempotent. Existing matching files are kept, while a
    same-name file with different content is reported as a conflict and left
    untouched. Set ``GIFT_HELPER_SKIP_EXAMPLE_ASSETS=1`` to opt out.
    """

    if os.environ.get("GIFT_HELPER_SKIP_EXAMPLE_ASSETS") == "1":
        return {name: "skipped" for name in EXAMPLE_ASSET_MANIFEST}

    if input_dir is None:
        try:
            import folder_paths
            input_dir = folder_paths.get_input_directory()
        except Exception as exc:
            LOGGER.warning("GiftHelperSuite could not resolve the ComfyUI input directory: %s", exc)
            return {name: "unavailable" for name in EXAMPLE_ASSET_MANIFEST}

    try:
        destination_dir = Path(input_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, TypeError) as exc:
        LOGGER.warning("GiftHelperSuite could not prepare the ComfyUI input directory: %s", exc)
        return {name: "error" for name in EXAMPLE_ASSET_MANIFEST}

    results: dict[str, str] = {}
    for filename, metadata in EXAMPLE_ASSET_MANIFEST.items():
        source = EXAMPLE_ASSET_DIR / filename
        destination = destination_dir / filename

        if not _matches_manifest(source, metadata):
            LOGGER.warning("GiftHelperSuite example asset is missing or invalid: %s", source)
            results[filename] = "invalid_source"
            continue

        if destination.exists():
            if _matches_manifest(destination, metadata):
                results[filename] = "present"
            else:
                LOGGER.warning(
                    "GiftHelperSuite kept an existing input file with different content: %s",
                    destination,
                )
                results[filename] = "conflict"
            continue

        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination_dir,
                prefix=f".{filename}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as temporary_handle:
                with source.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, temporary_handle, length=1024 * 1024)
                    temporary_handle.flush()
                    os.fsync(temporary_handle.fileno())

            if not _matches_manifest(temporary_path, metadata):
                raise OSError("temporary copy failed manifest verification")

            # A hard link publishes the fully verified file atomically and
            # fails instead of overwriting if another file already exists.
            os.link(temporary_path, destination)
        except FileExistsError:
            results[filename] = "present" if _matches_manifest(destination, metadata) else "conflict"
            continue
        except OSError as exc:
            LOGGER.warning("GiftHelperSuite could not install example asset %s: %s", filename, exc)
            results[filename] = "error"
            continue
        finally:
            if temporary_path is not None:
                _safe_unlink(temporary_path)

        if _matches_manifest(destination, metadata):
            LOGGER.info("GiftHelperSuite installed example asset: %s", destination)
            results[filename] = "installed"
        else:
            LOGGER.warning("GiftHelperSuite example asset failed verification: %s", destination)
            results[filename] = "error"

    return results


__all__ = ["EXAMPLE_ASSET_DIR", "EXAMPLE_ASSET_MANIFEST", "install_example_assets"]
