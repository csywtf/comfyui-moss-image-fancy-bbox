// Saved-workflow migration for the main node's widget values. Pure, so it is tested headlessly.
//
// ComfyUI stores a node's widget values as a positional list. Workflows saved before vae_downscale and
// patch_size were removed (the grid is now read from the latent) hold two extra values at indices 3-4:
//   old: width, height, regions, vae_downscale, patch_size, ramp_start, ramp_end, steps, strict, caption, ...
//   new: width, height, regions, ramp_start, ramp_end, steps, strict, caption, ...
// Every saved version has had the boolean `strict` at index 8 (old) or 6 (new), and index 8 is never a
// boolean in the new order, so that tells the two apart.

export function dropRemovedGridValues(values) {
  if (!Array.isArray(values) || values.length < 9) return values;
  if (typeof values[8] !== "boolean" || typeof values[6] === "boolean") return values;
  return [...values.slice(0, 3), ...values.slice(5)];
}
