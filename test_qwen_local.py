"""Legacy path for scripts.diagnostics.legacy.test_qwen_local."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("scripts.diagnostics.legacy.test_qwen_local", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("scripts.diagnostics.legacy.test_qwen_local")
