"""Legacy path for experiments.a0_legacy.train_fusiformer_production_editor_v5."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("experiments.a0_legacy.train_fusiformer_production_editor_v5", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("experiments.a0_legacy.train_fusiformer_production_editor_v5")
