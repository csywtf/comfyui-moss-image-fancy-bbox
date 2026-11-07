"""ComfyUI entry point for the MOSS-Image Regional LoRA custom node."""

WEB_DIRECTORY = "./web"

if __package__:  # ComfyUI imports this directory as a package; test runners may import the file bare
    from .moss_regional_lora.node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
