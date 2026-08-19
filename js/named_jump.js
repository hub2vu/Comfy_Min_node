import { app } from "../../scripts/app.js";


const SET_NODE_ID = "MinNamedSet";
const GET_NODE_ID = "MinNamedGet";


function escapeRegex(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}


function channelLabel(channel) {
    return channel ? `MIN GET: ${channel}` : "MIN GET: (unset)";
}


function channelRegex(channel) {
    return channel ? `^${escapeRegex(channelLabel(channel))}$` : "a^";
}


function ensureProperties(node) {
    node.properties ??= {};
    node.properties.ue_properties ??= {};
    Object.assign(node.properties.ue_properties, {
        version: "7.8",
        widget_ue_connectable: node.properties.ue_properties.widget_ue_connectable || {},
        input_ue_unconnectable: node.properties.ue_properties.input_ue_unconnectable || {},
    });
}


function channelWidget(node) {
    return node.widgets?.find((item) => item.name === "channel");
}


function syncSet(node) {
    ensureProperties(node);
    const channel = channelWidget(node)?.value?.trim() || "";
    node.properties.ue_convert = true;
    node.properties.rejects_ue_links = true;
    Object.assign(node.properties.ue_properties, {
        input_regex: channelRegex(channel),
        input_regex_invert: false,
        priority: 120,
    });
    app.canvas?.setDirty(true, true);
}


function syncGet(node) {
    ensureProperties(node);
    const channel = channelWidget(node)?.value?.trim() || "";
    const valueInput = node.inputs?.find((input) => input.name === "value");
    if (valueInput) valueInput.label = channelLabel(channel);
    node.properties.rejects_ue_links = false;
    node.properties.ue_properties.input_ue_unconnectable.value = false;
    app.canvas?.setDirty(true, true);
}


function setupNode(node, sync) {
    const widget = channelWidget(node);
    if (!widget || widget.minNodeJumpConfigured) {
        sync(node);
        return;
    }

    const originalCallback = widget.callback;
    widget.callback = function(value) {
        originalCallback?.apply(this, arguments);
        sync(node);
    };
    widget.minNodeJumpConfigured = true;
    sync(node);
}


function setupMatchingNode(node) {
    if (node.comfyClass === SET_NODE_ID) setupNode(node, syncSet);
    if (node.comfyClass === GET_NODE_ID) setupNode(node, syncGet);
}


app.registerExtension({
    name: "Min_node.NamedJump",
    nodeCreated: setupMatchingNode,
    loadedGraphNode: setupMatchingNode,
});
