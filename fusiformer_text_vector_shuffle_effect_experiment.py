"""Legacy path for experiments.text_effect.fusiformer_text_vector_shuffle_effect_experiment."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("experiments.text_effect.fusiformer_text_vector_shuffle_effect_experiment", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("experiments.text_effect.fusiformer_text_vector_shuffle_effect_experiment")
