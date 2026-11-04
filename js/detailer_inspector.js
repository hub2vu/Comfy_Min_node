import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


const STAGES = {
    MinFaceDetect: "detect", MinSegmentationDetect: "detect", MinFaceMask: "mask", MinFaceCropUpscale: "crop",
    MinFaceSelect: "select", MinDetailerWildcardPlan: "wildcard", MinFaceComposite: "composite",
};

const style = document.createElement("style");
style.textContent = `
.min-detailer-inspector { box-sizing:border-box; color:#e8ebef; background:#202329; border:1px solid #46505d; border-radius:7px; padding:10px; font:12px/1.5 sans-serif; overflow:auto; }
.min-detailer-inspector * { box-sizing:border-box; }
.min-detailer-inspector h4 { margin:2px 0 8px; font-size:13px; }
.min-detailer-inspector button { cursor:pointer; background:#354352; color:#fff; border:1px solid #63758a; border-radius:4px; padding:4px 9px; }
.min-detailer-toolbar { display:flex; gap:5px; flex-wrap:wrap; margin:6px 0; }
.min-detailer-note { color:#b6c3d0; margin:4px 0 8px; }
.min-detailer-card { border:1px solid #48515e; border-radius:5px; padding:8px; margin:8px 0; background:#282d35; }
.min-detailer-card[data-selected="false"] { border-color:#62666e; }
.min-detailer-card label { display:flex; align-items:center; gap:6px; cursor:pointer; font-weight:bold; }
.min-detailer-card input[type="checkbox"] { accent-color:#58d5bd; width:17px; height:17px; flex-shrink:0; }
.min-detailer-pair { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; margin-top:6px; }
.min-detailer-pair figure { min-width:0; margin:0; }
.min-detailer-overview .min-detailer-pair > figure:only-child { grid-column:1 / -1; }
.min-detailer-pair img, .min-detailer-overview img { width:100%; max-height:230px; object-fit:contain; background:#111; border-radius:3px; }
.min-detailer-pair figcaption { color:#bac9d8; font-size:11px; }
.min-detailer-inspector a { color:#88dfcc; }
.min-detailer-meta { overflow-wrap:anywhere; color:#c7d1dd; font-size:11px; margin-top:5px; }
.min-detailer-inspector summary { cursor:pointer; color:#a6ddcf; }
.min-detailer-inspector pre { white-space:pre-wrap; overflow-wrap:anywhere; font-size:10px; }
`;
document.head.appendChild(style);


function element(tag, text, className) {
    const item = document.createElement(tag);
    if (text != null) item.textContent = text;
    if (className) item.className = className;
    return item;
}

function imageUrl(file) {
    const query = new URLSearchParams({ filename: file.filename, subfolder: file.subfolder || "", type: file.type || "temp" });
    return api.apiURL(`/view?${query}`);
}

function picture(file, caption) {
    const figure = element("figure");
    figure.append(element("figcaption", caption));
    if (file) {
        const link = element("a");
        link.href = imageUrl(file);
        link.target = "_blank";
        link.rel = "noopener";
        const image = element("img");
        image.src = link.href;
        image.alt = caption;
        image.loading = "lazy";
        link.append(image);
        figure.append(link);
    } else {
        figure.append(element("div", "아직 처리된 이미지가 없습니다.", "min-detailer-note"));
    }
    return figure;
}

function selected(text, row) {
    text = String(text || "all").trim().toLowerCase();
    if (text === "all") return true;
    if (text === "none") return false;
    return text.split(/[,;\s]+/).some((token) => {
        const match = token.match(/^(?:(\d+):)?(\d+)(?:-(\d+))?$/);
        if (!match) return false;
        const first = Number(match[2]);
        const last = Number(match[3] ?? match[2]);
        return (match[1] == null || Number(match[1]) === row.image_index)
            && Math.min(first, last) <= row.index && row.index <= Math.max(first, last);
    });
}

function selectionWidget(node) {
    return node.widgets?.find((widget) => widget.name === "selected_indices");
}

function writeSelection(node, value) {
    const widget = selectionWidget(node);
    if (!widget) return;
    widget.value = value;
    widget.callback?.(value);
    node.graph?.change();
    app.canvas?.setDirty(true, true);
    render(node, node.minDetailerView);
}

function button(text, callback) {
    const item = element("button", text);
    item.type = "button";
    item.addEventListener("click", callback);
    return item;
}

function render(node, view) {
    const container = node.minDetailerElement;
    if (!container) return;
    const scrollTop = container.scrollTop;
    container.replaceChildren();
    const stage = STAGES[node.comfyClass];
    container.dataset.stage = stage;
    if (!view) {
        container.append(element("div", "실행하면 검출 위치, 마스크와 단계별 이미지가 표시됩니다.", "min-detailer-note"));
        return;
    }
    const rows = view.rows || [];
    const labels = { detect: "검출 위치 / class / 번호", mask: "마스크: 흰색=대상, 검정=바깥", crop: "원본 crop → 실제 확대 입력", select: "재생성할 영역 선택", wildcard: "Wildcard / 처리 순서 / seed", composite: "영역 합성 결과" };
    container.append(element("h4", `${labels[stage]} · ${rows.length}개`));
    if (stage === "select") {
        container.append(element("div", "체크한 영역만 기존 샘플러로 보냅니다. 번호는 이미지:검출 인덱스이며 0부터 시작합니다. 변경 후 다시 실행하세요.", "min-detailer-note"));
        const toolbar = element("div", null, "min-detailer-toolbar");
        toolbar.append(button("전체 선택", () => writeSelection(node, "all")), button("선택 해제", () => writeSelection(node, "none")));
        for (const label of new Set(rows.map((row) => row.class_label))) {
            toolbar.append(button(`class: ${label}`, () => writeSelection(node, rows.filter((row) => row.class_label === label).map((row) => row.id).join(",") || "none")));
        }
        container.append(toolbar);
    }
    for (const overview of view.overviews || []) {
        const panel = element("div", null, "min-detailer-overview");
        panel.append(element("div", `이미지 ${overview.image_index}${overview.mask_kind ? ` · ${overview.mask_kind}` : ""}`));
        const pair = element("div", null, "min-detailer-pair");
        pair.append(picture(overview.image, stage === "mask" ? "전체 마스크" : "검출 위치"));
        if (overview.overlay) pair.append(picture(overview.overlay, "원본 위 마스크"));
        panel.append(pair);
        container.append(panel);
    }
    for (const row of rows) {
        const card = element("section", null, "min-detailer-card");
        card.dataset.region = row.id;
        const title = `${row.id} · ${row.class_label} · ${Number(row.confidence).toFixed(3)}`;
        if (stage === "select") {
            const label = element("label");
            const checkbox = element("input");
            checkbox.type = "checkbox";
            checkbox.checked = selected(selectionWidget(node)?.value, row);
            checkbox.setAttribute("aria-label", `Process region ${row.id} (${row.class_label})`);
            card.dataset.selected = String(checkbox.checked);
            checkbox.addEventListener("change", () => {
                const chosen = new Set(rows.filter((item) => selected(selectionWidget(node)?.value, item)).map((item) => item.id));
                if (checkbox.checked) chosen.add(row.id); else chosen.delete(row.id);
                writeSelection(node, [...chosen].join(",") || "none");
            });
            label.append(checkbox, element("span", title));
            card.append(label);
        } else {
            card.append(element("strong", title));
        }
        const metadata = [`bbox: [${row.bbox.join(", ")}]`, `crop: [${row.crop_region.join(", ")}]`];
        if (row.source_detection_index != null) metadata.push(`검출 번호: ${row.source_detection_index}`);
        if (row.processing_order != null) metadata.push(`처리 순서: ${row.processing_order}`);
        if (row.seed != null) metadata.push(`seed: ${row.seed}`);
        if (row.text) metadata.push(`wildcard: ${row.text}`);
        if (row.final_crop_index != null) metadata.push(`최종 crop 출력: ${row.final_crop_index}`);
        if (row.status) metadata.push(row.status);
        card.append(element("div", metadata.join(" · "), "min-detailer-meta"));
        const pair = element("div", null, "min-detailer-pair");
        if (stage === "mask") pair.append(picture(row.mask, row.mask_kind), picture(row.region_preview, "대응 영역"));
        if (stage === "crop") pair.append(picture(row.cropped || row.region_preview, row.cropped ? "실제 원본 crop" : "검출 영역 (처리 전)"), picture(row.upscaled, "실제 확대 이미지"));
        if (stage === "select") pair.append(picture(row.region_preview, "선택 가능한 영역"));
        if (stage === "composite") pair.append(picture(row.final_refined || row.refined, "리사이즈·합성된 영역"));
        if (pair.children.length) card.append(pair);
        container.append(card);
    }
    const details = element("details");
    details.append(element("summary", "전체 좌표·class·seed JSON"), element("pre", JSON.stringify(view, null, 2)));
    container.append(details);
    container.scrollTop = scrollTop;
}

function setup(node) {
    if (!STAGES[node.comfyClass] || node.minDetailerElement) return;
    const container = element("div", null, "min-detailer-inspector");
    const widget = node.addDOMWidget("min_detailer_inspector", "MIN_DETAILER_INSPECTOR", container, {
        serialize: false, getMinHeight: () => 180, getMaxHeight: () => 560,
    });
    widget.computeSize = (width) => [width, 360];
    node.minDetailerElement = container;
    node.minDetailerView = node.properties?.min_detailer_view;
    render(node, node.minDetailerView);
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], 430), Math.max(node.size[1], size[1])]);
    const selection = selectionWidget(node);
    if (selection) {
        const previous = selection.callback;
        selection.callback = function () {
            previous?.apply(this, arguments);
            render(node, node.minDetailerView);
        };
    }
}

function update(view, fallback) {
    if (!view) return;
    const node = app.graph?.getNodeById(view.node_id) || fallback;
    if (!node || !STAGES[node.comfyClass]) return;
    setup(node);
    if (node.minDetailerView?.run_id === view.run_id) {
        const rows = new Map(node.minDetailerView.rows.map(row => [row.id, row]));
        for (const row of view.rows) rows.set(row.id, row);
        const overviews = new Map(node.minDetailerView.overviews.map(item => [item.image_index, item]));
        for (const item of view.overviews) overviews.set(item.image_index, item);
        view = { ...view, rows: [...rows.values()], overviews: [...overviews.values()] };
    }
    node.properties ??= {};
    node.properties.min_detailer_view = view;
    node.minDetailerView = view;
    render(node, view);
    app.canvas?.setDirty(true, true);
}

app.registerExtension({
    name: "Min_node.DetailerInspector",
    nodeCreated: setup,
    loadedGraphNode(node) {
        setup(node);
        if (node.properties?.min_detailer_view) update(node.properties.min_detailer_view, node);
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!STAGES[nodeData.name]) return;
        const previous = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            previous?.apply(this, arguments);
            for (const view of output?.min_detailer || []) update(view, this);
        };
    },
});
