import torch

def debug_obj(name, obj, max_items=5):
    print(f"\n========== {name} ==========")

    if obj is None:
        print("None")
        return

    if isinstance(obj, dict):
        print(f"type: dict, keys: {list(obj.keys())}")
        for k, v in obj.items():
            debug_obj(f"{name}.{k}", v, max_items=max_items)
        return

    if isinstance(obj, (list, tuple)):
        print(f"type: {type(obj).__name__}, len: {len(obj)}")
        for i, v in enumerate(obj[:max_items]):
            debug_obj(f"{name}[{i}]", v, max_items=max_items)
        return

    if isinstance(obj, torch.Tensor):
        print(
            f"type: Tensor, shape: {tuple(obj.shape)}, "
            f"dtype: {obj.dtype}, device: {obj.device}, "
            f"requires_grad: {obj.requires_grad}"
        )
        return

    if hasattr(obj, "num_nodes") and hasattr(obj, "num_edges"):
        print(
            f"type: DGLGraph, nodes: {obj.num_nodes()}, "
            f"edges: {obj.num_edges()}, device: {obj.device}"
        )
        print(f"ndata keys: {list(obj.ndata.keys())}")
        print(f"edata keys: {list(obj.edata.keys())}")
        return

    print(f"type: {type(obj)}, value: {obj}")