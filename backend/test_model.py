import sys
sys.path.insert(0, '.')

from model import MultiPathFusionNet
import torch

# Test model architecture with correct 1-channel input, 2-class output
model = MultiPathFusionNet(in_channels=1, num_classes=2)
model.eval()
x = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    y = model(x)
params = sum(p.numel() for p in model.parameters())
print('Model OK')
print('  Input  shape:', tuple(x.shape))
print('  Output shape:', tuple(y.shape))
print('  Parameters  :', f'{params:,}')

# Verify deterministic output
with torch.no_grad():
    y2 = model(x)
assert torch.equal(y, y2), "Model output is not deterministic!"
print('  Deterministic: ✓')

print('ALL TESTS PASSED')
