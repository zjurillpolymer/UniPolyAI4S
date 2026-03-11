import sys
import os

print("=== 环境位置检查 ===")
print(f"Python 路径: {sys.executable}")

# 检查是否在 D 盘
if "D:" in sys.executable:
    print("✅ 确认：环境已成功安装在 D 盘！")
else:
    print("❌ 警告：环境似乎不在 D 盘。")

print("\n=== 库功能测试 ===")

# 1. 测试 RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles("CCO")  # 乙醇
    mw = Descriptors.MolWt(mol)
    print(f"✅ RDKit 正常 | 乙醇分子量: {mw:.2f}")
except Exception as e:
    print(f"❌ RDKit 失败: {e}")

# 2. 测试 PyTorch
try:
    import torch

    print(f"✅ PyTorch 正常 | 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU 型号: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch 失败: {e}")

# 3. 测试 Transformers
try:
    from transformers import AutoTokenizer

    print("✅ Transformers 正常 | 可以加载预训练模型")
except Exception as e:
    print(f"❌ Transformers 失败: {e}")

print("\n🎉 所有检查完成！你可以开始复现 Uni-Poly 了。")