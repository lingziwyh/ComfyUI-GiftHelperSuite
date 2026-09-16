"""Automatic shot splitting and sparse-corrected MatAnyone2 video masks."""
from pathlib import Path
import importlib
import json
import time

import torch
import torch.nn.functional as F

from .matting_models import ensure_model
from .vendor.transnetv2 import TransNetV2


def matanyone_backend():
    import nodes

    node = nodes.NODE_CLASS_MAPPINGS.get("MatAnyone2")
    if node is None:
        raise RuntimeError(
            "Gift Adaptive Matting requires FuouM/ComfyUI-MatAnyone with MatAnyone2 support. "
            "Install/update that node pack and its requirements using your ComfyUI Python, "
            "then restart ComfyUI. Model weights download automatically when this node runs."
        )
    package = node.__module__.rsplit(".", 1)[0]
    backend = importlib.import_module(package + ".mat_anyone2")
    if not all(hasattr(backend, name) for name in ("get_matanyone2_model", "InferenceCore", "warming_up2")):
        raise RuntimeError("Update FuouM/ComfyUI-MatAnyone: incompatible MatAnyone2 backend")
    return backend


class GiftAutoShotSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "cut_threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
            "min_scene_frames": ("INT", {"default": 8, "min": 1, "max": 240}),
        }, "optional": {"auto_download": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("shots", "initial_frames", "shot_report")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "split"
    CATEGORY = "GiftHelperSuite/Video Matting"

    def split(self, images, cut_threshold, min_scene_frames, auto_download=True):
        import folder_paths
        import comfy.model_management

        started = time.perf_counter()
        if len(images) == 0:
            raise ValueError("Auto Shot Split requires at least one frame")
        checkpoint = ensure_model(
            "transnetv2", Path(folder_paths.models_dir), auto_download=auto_download,
            cancelled=comfy.model_management.throw_exception_if_processing_interrupted,
        )
        device = comfy.model_management.get_torch_device()
        net = TransNetV2().eval().to(device)
        net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        small = F.interpolate(images[..., :3].permute(0, 3, 1, 2), (27, 48), mode="bilinear", align_corners=False)
        frames = (small.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8)
        padded = torch.cat([frames[:1].repeat(25, 1, 1, 1), frames, frames[-1:].repeat(75, 1, 1, 1)])
        scores = []
        for start in range(0, len(images), 50):
            comfy.model_management.throw_exception_if_processing_interrupted()
            logits, _ = net(padded[start:start + 100].unsqueeze(0).to(device))
            scores.extend(logits.sigmoid()[0, 25:75, 0].tolist())
        scores = scores[:len(images)]
        runs = []
        for i, score in enumerate(scores):
            if score >= cut_threshold:
                if runs and i == runs[-1][-1] + 1:
                    runs[-1].append(i)
                else:
                    runs.append([i])
        deltas = [0.0] + (small[1:] - small[:-1]).abs().mean(dim=(1, 2, 3)).tolist()
        cuts = [0]
        for run in runs:
            peak = max(run, key=lambda i: scores[i])
            # Align a hard-cut candidate to the strongest adjacent RGB change.
            lo, hi = max(1, peak - 2), min(len(images), peak + 4)
            if lo >= hi:
                continue
            cut = max(range(lo, hi), key=lambda i: deltas[i])
            if cut - cuts[-1] >= min_scene_frames and len(images) - cut >= min_scene_frames:
                cuts.append(cut)
        cuts = sorted(set(cuts))
        cuts.append(len(images))
        ranges = list(zip(cuts[:-1], cuts[1:]))
        report = json.dumps({"frames": len(images), "ranges": ranges, "method": "TransNetV2 + local hard-cut alignment", "seconds": round(time.perf_counter() - started, 3)}, ensure_ascii=False)
        print("[GiftAutoShotSplit] " + report)
        return ([images[a:b].clone() for a, b in ranges], [images[a:a+1].clone() for a, b in ranges], report)


def checkpoint_indices(length, interval):
    if length < 1 or interval < 1:
        raise ValueError("Check Frames requires a nonempty video and a positive interval")
    indices = list(range(0, length, interval))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def backfill_weight(frame, left, right):
    if left == 0:
        return 1.0
    position = (frame - left) / (right - left)
    blend = min(1.0, max(0.0, (position - 0.25) / 0.5))
    return blend * blend * (3.0 - 2.0 * blend)


def disagreement_score(predicted, fresh, focus_ratio):
    height = max(1, round(predicted.shape[-2] * focus_ratio))
    pair = torch.stack([predicted[:height], fresh[:height]]).unsqueeze(1)
    pair = F.interpolate(pair, (192, 256), mode="area")[:, 0]
    # Ignore soft boundaries; compare confident interior labels only.
    mismatch = ((pair[0] < 0.2) & (pair[1] > 0.9)) | ((pair[0] > 0.8) & (pair[1] < 0.1))
    mismatch = -F.max_pool2d(-mismatch.float()[None, None], 3, 1, 1)[0, 0]
    union = ((pair[0] > 0.5) | (pair[1] > 0.5)).sum().item()
    return mismatch.sum().item() / max(union, 1)


class GiftMaskCheckFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "interval": ("INT", {"default": 12, "min": 1, "max": 240})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("check_frames", "check_indices")
    FUNCTION = "sample"
    CATEGORY = "GiftHelperSuite/Video Matting"

    def sample(self, images, interval):
        indices = checkpoint_indices(len(images), interval)
        return images[indices].clone(), json.dumps(indices)


class GiftAdaptiveMatting:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Invalidate Comfy's output cache after algorithm-only hot reloads.
        return "anchor_blend_v2"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "check_masks": ("MASK",), "check_indices": ("STRING", {"forceInput": True}),
            "disagreement_threshold": ("FLOAT", {"default": 0.025, "min": 0.001, "max": 1.0, "step": 0.005}),
            "focus_ratio": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05}),
            "n_warmup": ("INT", {"default": 10, "min": 1, "max": 30}),
            "max_internal_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
        }, "optional": {"auto_download": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("alpha", "correction_report")
    FUNCTION = "matte"
    CATEGORY = "GiftHelperSuite/Video Matting"

    def matte(self, images, check_masks, check_indices, disagreement_threshold, focus_ratio, n_warmup, max_internal_size, auto_download=True):
        import folder_paths
        import comfy.model_management
        import comfy.utils

        started = time.perf_counter()
        indices = json.loads(check_indices)
        if (not isinstance(indices, list) or not indices
                or any(type(i) is not int for i in indices)
                or indices[0] != 0 or indices != sorted(set(indices)) or indices[-1] >= len(images)):
            raise ValueError("check_indices must start at 0 and contain increasing frame indices within this shot")
        if len(indices) != len(check_masks) or tuple(check_masks.shape[1:]) != tuple(images.shape[1:3]):
            raise ValueError("One full-resolution check mask is required for each check index")
        backend = matanyone_backend()
        legacy = Path(backend.__file__).parent / "checkpoint" / "matanyone2.pth"
        checkpoint = ensure_model(
            "matanyone2", Path(folder_paths.models_dir), extra_paths=(legacy,),
            auto_download=auto_download,
            cancelled=comfy.model_management.throw_exception_if_processing_interrupted,
        )
        model = backend.get_matanyone2_model(str(checkpoint), comfy.model_management.get_torch_device())
        model.cfg.max_internal_size = max_internal_size
        model.cfg.max_mem_frames = 5
        model.cfg.use_long_term = False
        frames = images[..., :3].permute(0, 3, 1, 2)
        checkpoints = dict(zip(indices, check_masks))
        outputs = [None] * len(images)
        checks = []
        progress = comfy.utils.ProgressBar(len(images))

        def initialize(index):
            core = backend.InferenceCore(model, cfg=model.cfg)
            seed = (checkpoints[index].float().clamp(0, 1) * 255).to(torch.uint8).float().to(core.device)
            repeated = frames[index:index+1].repeat(n_warmup, 1, 1, 1).to(core.device)
            probability = backend.warming_up2(seed, core, n_warmup, repeated)
            return core, core.output_prob_to_mask(probability).cpu().clamp(0, 1)

        processor, outputs[0] = initialize(0)
        progress.update(1)
        previous_check = 0
        for index in range(1, len(images)):
            comfy.model_management.throw_exception_if_processing_interrupted()
            probability = processor.step(frames[index].to(processor.device))
            predicted = processor.output_prob_to_mask(probability).cpu().clamp(0, 1)
            if index in checkpoints:
                score = disagreement_score(predicted, checkpoints[index].cpu(), focus_ratio)
                corrected = score >= disagreement_threshold
                checks.append({"frame": index, "disagreement": round(score, 5), "reset": corrected,
                               "backfill_start": previous_check if corrected else None})
                if corrected:
                    processor, predicted = initialize(index)
                    backward, _ = initialize(index)
                    for earlier in range(index - 1, previous_check - 1, -1):
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        probability_back = backward.step(frames[earlier].to(backward.device))
                        revised = backward.output_prob_to_mask(probability_back).cpu().clamp(0, 1)
                        # Preserve the left anchor; favor the nearest temporal direction.
                        weight = backfill_weight(earlier, previous_check, index)
                        outputs[earlier] = outputs[earlier] * (1 - weight) + revised * weight
                    del backward
                previous_check = index
            outputs[index] = predicted
            progress.update(1)
        report = json.dumps({"frames": len(images), "check_indices": indices, "checks": checks,
                             "reset_count": sum(c["reset"] for c in checks), "seconds": round(time.perf_counter()-started, 3)}, ensure_ascii=False)
        print("[GiftAdaptiveMatting] " + report)
        return torch.stack(outputs), report

NODE_CLASS_MAPPINGS = {
    "GiftAutoShotSplit": GiftAutoShotSplit,
    "GiftMaskCheckFrames": GiftMaskCheckFrames,
    "GiftAdaptiveMatting": GiftAdaptiveMatting,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GiftAutoShotSplit": "Gift Auto Shot Split | 自动切镜",
    "GiftMaskCheckFrames": "Gift Mask Check Frames | 遮罩检查帧",
    "GiftAdaptiveMatting": "Gift Adaptive Matting | 自适应视频抠像",
}
