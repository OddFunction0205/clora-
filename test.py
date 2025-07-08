import os
from safetensors import safe_open

# 修改为你模型文件所在目录
model_dir = "/root/autodl-tmp/model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"

# 遍历所有 .safetensors 文件
for filename in sorted(os.listdir(model_dir)):
    if filename.endswith(".safetensors"):
        file_path = os.path.join(model_dir, filename)
        print(f"🔍 正在检查文件：{filename}")
        try:
            with safe_open(file_path, framework="pt") as f:
                keys = f.keys()
            print(f"✅ {filename} 正常，包含 {len(keys)} 个权重键。")
        except Exception as e:
            print(f"❌ {filename} 损坏或无法打开：{e}")
