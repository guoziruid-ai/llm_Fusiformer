"""Legacy path for scripts.experiments.qwen_legacy.main_crysformer."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("scripts.experiments.qwen_legacy.main_crysformer", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("scripts.experiments.qwen_legacy.main_crysformer")
