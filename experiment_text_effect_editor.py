"""Legacy path for experiments.text_effect.experiment_text_effect_editor."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("experiments.text_effect.experiment_text_effect_editor", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("experiments.text_effect.experiment_text_effect_editor")
