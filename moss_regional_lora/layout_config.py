"""MOSS-Image sequence layout, as established on real hardware by tools/discover_moss_layout.py.

Every value here must come from running the discovery tool against your checkpoint.
Do not fill these in by hand from documentation or intuition: a wrong value
produces a node that *looks* like it works (the LoRA still changes the image) while
the mask lands on the wrong tokens.

While any value is None the node refuses to run when ``strict`` is on (FR-11).

TEXT_POSITION
    "prefix"  -> the transformer sequence is [text tokens][image tokens]...
    "suffix"  -> the transformer sequence is [image tokens][text tokens]...

IMAGE_ORDER
    "raster"  -> image token k is the patch at row k // cols, column k % cols,
                 row 0 at the top of the image, column 0 at the left.
    (No other order is implemented. If discovery finds a different order, the
    tool says so and the node must not be used until the order is implemented.)

SOURCE_TOKENS
    Where an edit pipeline's source-image tokens sit in the sequence, if at all:
    "none"    -> source images add no tokens to the sequence
    "start"   -> [source][...text and image as above...]
    "between" -> between the text block and the image block
    "end"     -> after both the text and the image blocks
    Source tokens never map to an output position and are always masked to zero.
"""

TEXT_POSITION = None
IMAGE_ORDER = None
SOURCE_TOKENS = None
