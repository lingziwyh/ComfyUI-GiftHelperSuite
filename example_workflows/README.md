# GiftHelperSuite example workflows

Drag either JSON file into ComfyUI after installing GiftHelperSuite and restarting ComfyUI:

- `Gift_PostFX_Example.json`
- `Gift_Chroma_Master_Example.json`
- `Gift_Icon_Auto_Restore_Production.json`

The `assets` folder is part of the plugin package. During startup, GiftHelperSuite installs the three uniquely named demo assets into the root of `ComfyUI/input` without overwriting existing files. Both workflows use only input-relative filenames and contain no developer-machine paths or temporary preview files.

The video loader and encoder nodes come from `ComfyUI-VideoHelperSuite`. The GiftHelperSuite nodes and example media require no model downloads.

`Gift_Icon_Auto_Restore_Production.json` is the full production template for the Klein,
chroma-key, transparent-effect recovery, and three-output ICON chain. It expects the
Flux Klein model files named in the loader nodes. Upload the intended black-background
source image in place of `GiftHelperSuite_Icon_Source.png`; its preview guide background
is bundled inside `GiftIconAutoRestore`, so no preview-background loader is needed.

安装组件并重启 ComfyUI 后，可直接把上面的 JSON 拖入界面。组件会把 `assets` 中的三个示例素材安全复制到 `ComfyUI/input` 根目录，遇到同名文件时不会覆盖。两个工作流均已清理本机绝对路径和临时预览缓存。
