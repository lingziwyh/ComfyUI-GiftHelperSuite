"""Pinned, integrity-checked downloads, only when a matting node executes."""
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import tempfile
import threading
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ModelAsset:
    filename: str
    url: str
    sha256: str
    legacy_paths: tuple = ()


MODELS = {
    "transnetv2": ModelAsset(
        "transnetv2-pytorch-weights.pth",
        "https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights/resolve/"
        "a97542e4eb22e3af904ac13b10cf06da507e2ff1/transnetv2-pytorch-weights.pth",
        "46520d66d4bf60414a4d82e0e94a92442ff950e34517a3718b2e54815e642b53",
        ("VLM/transnetv2-pytorch-weights/transnetv2-pytorch-weights.pth",),
    ),
    "matanyone2": ModelAsset(
        "matanyone2.pth",
        "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth",
        "5e9821e4087231427376b437c85bb6e072b41e582314f06fd524f75bc4af5914",
        ("MatAnyone2/matanyone2.pth", "matanyone2/matanyone2.pth"),
    ),
}
_LOCK = threading.Lock()
_LOG = logging.getLogger(__name__)


def _check_hash(path, expected, cancelled):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            cancelled()
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(
            f"Model integrity check failed: {path}. The file was NOT overwritten. "
            "Move it aside and retry, or manually install the documented checkpoint."
        )


def ensure_model(name, models_dir, *, extra_paths=(), auto_download=True, cancelled=lambda: None):
    asset = MODELS[name]  # Fixed allowlist, never a user-supplied URL or filename.
    root = Path(models_dir)
    target = root / "GiftHelperSuite" / asset.filename
    candidates = [target, *(root / p for p in asset.legacy_paths), *map(Path, extra_paths)]
    with _LOCK:
        for path in candidates:
            if path.is_file():
                _check_hash(path, asset.sha256, cancelled)
                return path
        if not auto_download or os.environ.get("GIFT_HELPER_AUTO_DOWNLOAD", "1") == "0":
            raise FileNotFoundError(
                f"Missing {name} weights and automatic download is disabled. "
                f"Download {asset.url} and place the file at {target}"
            )
        cancelled()
        target.parent.mkdir(parents=True, exist_ok=True)
        _LOG.warning("GiftHelperSuite: downloading %s to %s", name, target)
        fd, temporary_name = tempfile.mkstemp(prefix=asset.filename + ".", suffix=".part", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as out:
                request = Request(asset.url, headers={"User-Agent": "GiftHelperSuite-model-download"})
                with urlopen(request, timeout=30) as response:
                    if response.status != 200:
                        raise OSError(f"Unexpected HTTP status: {response.status}")
                    while chunk := response.read(1024 * 1024):
                        cancelled()
                        out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            _check_hash(temporary, asset.sha256, cancelled)
            cancelled()
            # Hard-link publishes the complete file atomically without clobbering
            # a file another process may have installed while we downloaded.
            try:
                os.link(temporary, target)
            except FileExistsError:
                _check_hash(target, asset.sha256, cancelled)
            _LOG.warning("GiftHelperSuite: %s ready", name)
            return target
        except InterruptedError:
            raise
        except (OSError, URLError) as exc:
            raise RuntimeError(
                f"Could not download {name}. Check network/proxy and run again. "
                f"Manual URL: {asset.url}; destination: {target}. Reason: {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
