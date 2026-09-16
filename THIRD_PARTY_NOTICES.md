# Third-party components

## TransNetV2

`vendor/transnetv2.py` is copied from
[soCzech/TransNetV2](https://github.com/soCzech/TransNetV2), commit
`85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed`,
`inference-pytorch/transnetv2_pytorch.py`. It retains the MIT license in
`vendor/LICENSE.TransNetV2`. No TensorFlow or external shot-split plugin is needed.
The PyTorch checkpoint is hosted by
[MiaoshouAI](https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights), pinned
to revision `a97542e4eb22e3af904ac13b10cf06da507e2ff1` and verified by SHA-256.

## MatAnyone2 (external dependency)

The adapter uses [FuouM/ComfyUI-MatAnyone](https://github.com/FuouM/ComfyUI-MatAnyone),
tested at `87cbce38c03bb359471bd23704ebd1720e98a842` (2.1.4).
Neither its implementation nor model weights are redistributed here.
Missing weights are downloaded from the
[official MatAnyone2 v1.0.0 release](https://github.com/pq-yang/MatAnyone2/releases/tag/v1.0.0).
The checkpoint hash is pinned in `matting_models.py`.
[MatAnyone2](https://github.com/pq-yang/MatAnyone2) has its own
[NTU S-Lab License 1.0](https://github.com/pq-yang/MatAnyone2/blob/main/LICENSE.txt);
GiftHelperSuite's MIT license does not relicense the model or grant commercial use.

## Example workflow dependencies

Foreground masks come from
[ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG), which owns downloading
the selected InSPyReNet/RMBG model. Its code is not copied here.
Video I/O uses [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite).
Mask list batching uses [KJNodes](https://github.com/kijai/ComfyUI-KJNodes).
Those projects and their selected models retain their respective licenses.
