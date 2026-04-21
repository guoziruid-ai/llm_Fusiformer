import os 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from .utils import compute_angle
from transformers import AutoModel, AutoTokenizer


QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 

## Contrastive Transforms ##
class ContrastiveTransform(object):
    """Take two transforms of one crystal"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        x1 = self.base_transform1(x)
        x2 = self.base_transform2(x)
        return [x1, x2]

## Structure Transforms ##
class LoadfromStructure(object):
    """Load infos from crystal structure"""
    def __init__(self, infos=['atomic_numbers', 'cart_coords', 'lattice_mat', 'num_atoms']):
        self.infos = infos

    def __call__(self, crystal):
        for info in self.infos:
            crystal[info] = getattr(crystal['structure'], info)
        return crystal


class Padding(object):
    """Padd atoms to a fix number"""
    def __init__(self, max_atoms=300):
        self.max_atoms = max_atoms
        self.mode = {
            1: (1, 1, 1),
            # 2: (1, 1, 2),
            # 4: (1, 2, 2),
            8: (2, 2, 2),
            # 12: (2, 2, 3),
            # 18: (2, 3, 3),
            27: (3, 3, 3),
            # 36: (3, 3, 4),
            # 48: (3, 4, 4),
            64: (4, 4, 4),
            # 80: (4, 4, 5),
            125: (5, 5, 5),
            216: (6, 6, 6),
            343: (7, 7, 7)
        }

    def __call__(self, crystal):
        n = len(crystal['atomic_numbers'])
        for repeat in self.mode:
            if n * repeat > self.max_atoms:
                atomic_numbers = crystal['atomic_numbers']
                coords = crystal['cart_coords']
                lattice = crystal['lattice_mat']
                crystal['atomic_numbers'] = np.tile(atomic_numbers, repeat)
                crystal['cart_coords'] = self.pad_coords(coords, lattice, repeat)
                break
        return crystal

    def pad_coords(self, coords, lattice, n):
        i, j, k = self.mode[n]
        i, j, k = np.meshgrid(np.arange(i), np.arange(j), np.arange(k), indexing='ij')
        ijk = np.stack([i, j, k], 3).reshape(-1, 1, 3)
        offset = ijk @ lattice
        coords = offset + np.expand_dims(coords, 0)
        return coords.reshape(-1, 3)


class Rotating(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, crystal):
        rot_mat = R.random().as_matrix()
        crystal['cart_coords'] = crystal['cart_coords'] @ rot_mat
        crystal['lattice_mat'] = crystal['lattice_mat'] @ rot_mat
        return crystal


class Centering(object):
    def __init__(self):
        pass

    def __call__(self, crystal):
        coords = crystal['cart_coords']
        crystal['cart_coords'] = coords - coords.mean(0)
        return crystal

class CutOff(object):
    def __init__(self, max_atoms=300):
        self.max_atoms = max_atoms

    def __call__(self, crystal):
        coords = crystal['cart_coords']
        idx = np.linalg.norm(coords, axis=-1).argsort()
        idx = idx[:self.max_atoms]
        crystal['atomic_numbers'] = crystal['atomic_numbers'][idx]
        crystal['cart_coords'] = crystal['cart_coords'][idx]
        return crystal

class StructureCollecting(object):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __call__(self, crystal):
        inputs = []
        targets = []
        for i in self.inputs:
            assert i in crystal
            if i == 'atomic_numbers' or i == 'num_atoms':
                inputs.append(torch.tensor(crystal[i], dtype=torch.long))
            else:
                inputs.append(torch.tensor(crystal[i], dtype=torch.float32))
        for i in self.targets:
            if i in crystal:
                targets.append(torch.tensor([crystal[i]], dtype=torch.float32))
            else:
                assert i in crystal['info']
                targets.append(torch.tensor([crystal['info'][i]], dtype=torch.float32))
        return inputs, targets


## Graph Transforms ##
class AddNodeFeature(object):
    def __init__(self, features=['atomic_numbers'], use_qwen=True, qwen_model_name=QWEN_MODEL_NAME, freeze_qwen=True):
        self.features = features
        self.use_qwen = use_qwen
        self.qwen_model_name = qwen_model_name
        
        # 加载 Qwen 模型
        if use_qwen:
            # 设置环境变量使用国内镜像（如果网络有问题）
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            
            # Qwen 使用 AutoModel 加载
            self.qwen_model = AutoModel.from_pretrained(
                qwen_model_name,
                trust_remote_code=True  # Qwen 需要 trust_remote_code
            )
            if freeze_qwen:
                for param in self.qwen_model.parameters():
                    param.requires_grad = False
            self.qwen_model.eval()  # 设置为评估模式
        else:
            self.qwen_model = None

    def __call__(self, crystal):
        structure = crystal['structure']
        for feat in self.features:
            assert hasattr(structure, feat)
            crystal['graph'].ndata[feat] = torch.tensor(getattr(structure, feat), dtype=torch.long)
        
        # 处理文本特征 - 使用 Qwen
        if "text" in crystal and self.use_qwen and self.qwen_model is not None:
            text_inputs = crystal["text"]
            
            # 确保输入在正确的设备上
            device = next(self.qwen_model.parameters()).device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                # 使用 Qwen 编码文本
                outputs = self.qwen_model(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs.get('attention_mask', None),
                    output_hidden_states=True
                )
                
                # 使用 attention mask 进行加权平均池化
                attention_mask = text_inputs.get('attention_mask', None)
                if attention_mask is not None:
                    hidden_states = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    text_embedding = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
                else:
                    text_embedding = outputs.last_hidden_state.mean(dim=1)
            
            # 移回 CPU 以便后续处理
            crystal['text_emb'] = text_embedding.cpu()
        
        return crystal


class AddEdgeFeature(object):
    def __init__(self, features=['distance']):
        self.features = features

    def __call__(self, crystal):
        for feat in self.features:
            if feat == 'offset':
                pass
            elif feat == 'distance':
                crystal['graph'].edata[feat] = torch.norm(crystal['graph'].edata.pop('offset'), dim=1).float()
        return crystal


class AddAngleFeature(object):
    def __init__(self, features=['angle']):
        self.features = features
    
    def __call__(self, crystal):
        assert 'line_graph' in crystal
        crystal['line_graph'].apply_edges(compute_angle)
        crystal['line_graph'].ndata.pop('offset')
        return crystal


class GraphCollecting(object):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __call__(self, crystal):
        if self.inputs == ['graph', 'line_graph']:
            inputs_graph = [crystal['graph'], crystal['line_graph']]
        elif self.inputs == ['graph']:
            inputs_graph = [crystal['graph']]
        else:
            raise ValueError('Invalid inputs!')
        inputs = {'graph_input': inputs_graph}

        if 'text_emb' in crystal:
            inputs['text_emb'] = crystal['text_emb']

        targets = []
        for i in self.targets:
            if i in crystal:
                targets.append(torch.tensor([crystal[i]], dtype=torch.float32))
            else:
                assert i in crystal['info']
                targets.append(torch.tensor([crystal['info'][i]], dtype=torch.float32))
        return inputs, torch.cat(targets)


## Compose
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x