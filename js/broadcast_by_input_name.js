import { app } from "../../../scripts/app.js";


const NODE_ID = "MinBroadcastByInputName";


function escapeRegex(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}


function syncTargetName(node) {
    const widget = node.widgets?.find((item) => item.name === "target_input_name");
    const targetName = widget?.value?.trim() || "";

    node.properties ??= {};
    node.properties.ue_properties ??= {};
    node.properties.ue_convert = true;
    node.properties.rejects_ue_links = true;
    Object.assign(node.properties.ue_properties, {
        version: "7.8",
        input_regex: targetName ? `^${escapeRegex(targetName)}$` : "a^",
        input_regex_invert: false,
        group_restricted: 0,
        color_restricted: 0,
        widget_ue_connectable: node.properties.ue_properties.widget_ue_connectable || {},
        input_ue_unconnectable: node.properties.ue_properties.input_ue_unconnectable || {},
        priority: 100,
    });

    app.canvas?.setDirty(true, true);
}


function setupNode(node) {
    const widget = node.widgets?.find((item) => item.name === "target_input_name");
    if (!widget || widget.minNodeBroadcastConfigured) {
        syncTargetName(node);
        return;
    }

    const originalCallback = widget.callback;
    widget.callback = function(value) {
        originalCallback?.apply(this, arguments);
        syncTargetName(node);
    };
    widget.minNodeBroadcastConfigured = true;
    syncTargetName(node);
}


app.registerExtension({
    name: "Min_node.BroadcastByInputName",

    nodeCreated(node) {
        if (node.comfyClass === NODE_ID) setupNode(node);
    },

    loadedGraphNode(node) {
        if (node.comfyClass === NODE_ID) setupNode(node);
    },
});
