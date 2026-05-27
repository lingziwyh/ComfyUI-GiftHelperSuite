# Merge notes

Files added / changed:

```text
added:    ae_rgba_tools.py
modified: __init__.py
added:    README_AE_RGBA_TOOLS.md
```

New nodes:

```text
AE Unmult RGBA
AE Alpha Over RGBA
```

`AE Alpha Over RGBA` includes:

```text
blend_mode dropdown
foreground_opacity slider
background_opacity slider
resize_foreground_to_background option
```

Suggested git workflow:

```bash
cd ComfyUI/custom_nodes/ComfyUI-GiftHelperSuite
git checkout -b add-ae-rgba-tools
# copy ae_rgba_tools.py, README_AE_RGBA_TOOLS.md, and replace __init__.py
git add ae_rgba_tools.py README_AE_RGBA_TOOLS.md __init__.py
git commit -m "Add AE RGBA compositing tools"
git push origin add-ae-rgba-tools
```
