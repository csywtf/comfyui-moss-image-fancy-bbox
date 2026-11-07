// Headless tests for web/migrate.js. Run: node --test tests/web/
import { test } from "node:test";
import assert from "node:assert/strict";
import { dropRemovedGridValues } from "../../web/migrate.js";

const REGIONS = '{"version": 2, "regions": []}';

test("workflows saved with vae_downscale/patch_size lose those two values, every saved length", () => {
  const tail = [0.15, 1.0, 20, true, "a caption", "suffix", "raster", "end"];
  for (const keep of [4, 5, 8]) {              // no caption / caption / caption + layout options
    const old = [1024, 768, REGIONS, 16, 1, ...tail.slice(0, keep)];
    assert.deepEqual(dropRemovedGridValues(old), [1024, 768, REGIONS, ...tail.slice(0, keep)]);
  }
});

test("current workflows and anything unrecognised are left alone", () => {
  const current = [1024, 768, REGIONS, 0.15, 1.0, 20, true, "a caption", "suffix", "raster", "end"];
  assert.deepEqual(dropRemovedGridValues(current), current);
  assert.deepEqual(dropRemovedGridValues(current.slice(0, 8)), current.slice(0, 8));
  for (const v of [undefined, null, {}, [1, 2, 3]]) assert.equal(dropRemovedGridValues(v), v);
  // migrating twice is a no-op
  const old = [1024, 768, REGIONS, 16, 1, 0.15, 1.0, 20, false, ""];
  assert.deepEqual(dropRemovedGridValues(dropRemovedGridValues(old)), dropRemovedGridValues(old));
});
