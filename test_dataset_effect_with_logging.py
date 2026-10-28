"""Legacy path for experiments.qwen_legacy.test_dataset_effect_with_logging."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("experiments.qwen_legacy.test_dataset_effect_with_logging", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("experiments.qwen_legacy.test_dataset_effect_with_logging")
