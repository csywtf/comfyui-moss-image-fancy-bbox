// Shows the "MOSS-Image Layout Discovery" report on the node after it runs. The probe images are shown
// by ComfyUI's standard image preview (the node returns them as preview images).

import { app } from "../../scripts/app.js";

const NODE_NAME = "MossLayoutDiscovery";

app.registerExtension({
  name: "moss.LayoutDiscoveryReport",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const area = document.createElement("textarea");
      area.readOnly = true;
      area.placeholder = "Queue this node once. The report and the values to set appear here.";
      area.style.cssText =
        "width:100%;height:100%;box-sizing:border-box;font:11px monospace;white-space:pre;resize:none;" +
        "background:var(--comfy-input-bg,#222);color:var(--input-text,#ddd);border:1px solid var(--border-color,#555)";
      // let the text be selected and copied without dragging the node
      area.addEventListener("pointerdown", (e) => e.stopPropagation());
      area.addEventListener("keydown", (e) => e.stopPropagation());
      const widget = this.addDOMWidget("report", "moss_layout_report", area, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 220,
        getValue: () => undefined,
        setValue: () => {},
      });
      widget.serialize = false;
      this.__mossReport = area;
      this.setSize([Math.max(this.size[0], 560), Math.max(this.size[1], 520)]);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      const text = output?.moss_layout_report?.[0];
      if (this.__mossReport && typeof text === "string") {
        this.__mossReport.value = text;
        // the values to copy are at the end
        this.__mossReport.scrollTop = this.__mossReport.scrollHeight;
      }
    };
  },
});
