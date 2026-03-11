import torch
import torchdata
import dgl

print(f"PyTorch: {torch.__version__}")
print(f"TorchData: {torchdata.__version__}")
print(f"DGL: {dgl.__version__}")

# 终极测试：尝试创建一个 DGL 图
try:
    g = dgl.graph(([0, 1], [1, 2])).to('cuda')
    print("DGL GPU 测试成功！")
except Exception as e:
    print(f"DGL 依然不可用，错误原因: {e}")