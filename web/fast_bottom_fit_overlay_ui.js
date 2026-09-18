import { app } from "../../scripts/app.js";

const NODE_NAME = "FastBottomFitOverlay";
const PRESET_VALUES = {
    "Low Coins": {
        fade_mode: "Rounded Rect",
        background_fade_ratio: 0.0,
        rounded_rect_fade_ratio: 0.255,
        rounded_corner_radius: 0.90,
    },
    Standard: {
        fade_mode: "Top Fade",
        top_fade_ratio: 0.08,
        background_fade_ratio: 0.0,
    },
    "Hybrid Naked-Eye 3D": {
        fade_mode: "None",
        background_fade_ratio: 0.0,
    },
    "Naked-Eye 3D": {
        fade_mode: "Top Fade",
        top_fade_ratio: 0.045,
        background_fade_ratio: 0.52,
    },
};

const CONTROLLED_WIDGETS = new Set([
    "fade_mode",
    "top_fade_ratio",
    "background_fade_ratio",
    "rounded_rect_fade_ratio",
    "rounded_corner_radius",
]);

function isInactiveForMode(widgetName, fadeMode) {
    if (widgetName === "top_fade_ratio") return fadeMode !== "Top Fade";
    if (widgetName === "rounded_rect_fade_ratio" || widgetName === "rounded_corner_radius") {
        return fadeMode !== "Rounded Rect";
    }
    return false;
}

function captureCustomValues(node) {
    node._giftCustomValues = Object.fromEntries(
        (node.widgets ?? [])
            .filter((widget) => CONTROLLED_WIDGETS.has(widget.name))
            .map((widget) => [widget.name, widget.value]),
    );
}

function restoreCustomValues(node) {
    for (const widget of node.widgets ?? []) {
        if (Object.prototype.hasOwnProperty.call(node._giftCustomValues ?? {}, widget.name)) {
            widget.value = node._giftCustomValues[widget.name];
        }
    }
}

function captureEditableValues(node, presetName) {
    const values = PRESET_VALUES[presetName] ?? null;
    node._giftCustomValues ??= {};
    for (const widget of node.widgets ?? []) {
        if (!CONTROLLED_WIDGETS.has(widget.name)) continue;
        const presetControlled = values !== null
            && Object.prototype.hasOwnProperty.call(values, widget.name);
        if (!presetControlled) node._giftCustomValues[widget.name] = widget.value;
    }
}

function refreshWidgetState(node, presetName) {
    const values = PRESET_VALUES[presetName] ?? null;
    const fadeMode = node.widgets?.find((widget) => widget.name === "fade_mode")?.value ?? "None";
    for (const widget of node.widgets ?? []) {
        if (!CONTROLLED_WIDGETS.has(widget.name)) continue;
        const presetControlled = values !== null && Object.prototype.hasOwnProperty.call(values, widget.name);
        const inactive = isInactiveForMode(widget.name, fadeMode);
        widget.disabled = presetControlled || inactive;
        widget._giftBaseLabel ??= widget.label ?? widget.name;
        widget.label = presetControlled
            ? `🔒 ${widget._giftBaseLabel}`
            : (inactive ? `○ ${widget._giftBaseLabel}` : widget._giftBaseLabel);
    }
    node.setDirtyCanvas?.(true, true);
}

function syncPresetUI(node, presetName) {
    captureEditableValues(node, node._giftActivePreset ?? "Custom");
    restoreCustomValues(node);
    const values = PRESET_VALUES[presetName] ?? null;
    if (values !== null) {
        for (const widget of node.widgets ?? []) {
            if (Object.prototype.hasOwnProperty.call(values, widget.name)) {
                widget.value = values[widget.name];
            }
        }
    }
    node._giftActivePreset = presetName;
    refreshWidgetState(node, presetName);
}

app.registerExtension({
    name: "GiftHelperSuite.FastBottomFitOverlay.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);
            const preset = this.widgets?.find((widget) => widget.name === "preset");
            const fadeMode = this.widgets?.find((widget) => widget.name === "fade_mode");
            if (!preset || !fadeMode) return;

            captureCustomValues(this);
            this._giftActivePreset = "Custom";
            const originalCallback = preset.callback;
            preset.callback = (value, ...args) => {
                originalCallback?.call(preset, value, ...args);
                syncPresetUI(this, value);
            };

            const originalFadeModeCallback = fadeMode.callback;
            fadeMode.callback = (value, ...args) => {
                originalFadeModeCallback?.call(fadeMode, value, ...args);
                fadeMode.value = value;
                refreshWidgetState(this, preset.value);
            };
            syncPresetUI(this, preset.value);
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            originalConfigure?.apply(this, arguments);
            const preset = this.widgets?.find((widget) => widget.name === "preset");
            captureCustomValues(this);
            this._giftActivePreset = "Custom";
            syncPresetUI(this, preset?.value ?? "Custom");
        };
    },
});
