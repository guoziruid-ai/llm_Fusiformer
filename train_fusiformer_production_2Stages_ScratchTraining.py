"""Legacy path for the MatSciBERT two-stage scratch experiment."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("experiments.matscibert.train_fusiformer_production_2Stages_ScratchTraining", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("experiments.matscibert.train_fusiformer_production_2Stages_ScratchTraining")
