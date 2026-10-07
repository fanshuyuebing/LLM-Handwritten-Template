"""Smoke-check installed dependencies and every template without network access."""
import importlib.util
from pathlib import Path
import socket
import sys
import torch

def blocked(*args, **kwargs):
    raise RuntimeError('离线检查禁止网络连接')
socket.socket.connect = blocked
socket.create_connection = blocked
root = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
x = torch.randn(4, 8, requires_grad=True)
loss = x.square().mean()
loss.backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
for group in ('Transformer-EASY', 'Transformer-HARD', 'RLHF-EASY', 'RLHF-HARD', 'Interview-EASY', 'Interview-HARD'):
    sys.path.insert(0, str(root / group))
    sys.modules.pop('rlhf_env', None)
    for path in sorted((root / group).glob('*.py')):
        name = group.replace('-', '_') + '_' + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f'✓ 导入 {group}/{path.name}')
    sys.path.pop(0)
print(f'✓ Python {sys.version.split()[0]} / PyTorch {torch.__version__}')
print('✓ CPU 张量计算与反向传播通过；所有模板可离线导入。')
print('注意：算法 TODO 尚未实现，此检查不代表算法练习已通过。')
