"""Legacy path for the MatSciBERT weights-mode experiment."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("experiments.matscibert.train_fusiformer_production_WeghtsMode", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("experiments.matscibert.train_fusiformer_production_WeghtsMode")
