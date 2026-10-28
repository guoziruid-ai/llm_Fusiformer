"""Legacy path for experiments.text_effect.analyze_text_effect_results_robo_wiki."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("experiments.text_effect.analyze_text_effect_results_robo_wiki", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("experiments.text_effect.analyze_text_effect_results_robo_wiki")
