"""Legacy path for scripts.diagnostics.legacy.debug_dataloader_batch."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("scripts.diagnostics.legacy.debug_dataloader_batch", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("scripts.diagnostics.legacy.debug_dataloader_batch")
