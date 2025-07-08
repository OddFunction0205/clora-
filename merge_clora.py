import os
import torch
from transformers import AutoModelForCausalLM
from clora.modeling_clora import CLoRALinear

# 路径配置
BASE_MODEL_PATH = "/root/autodl-tmp/model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
CLORA_MODEL_PATH = "/root/autodl-tmp/clora_commonsense_model/checkpoint-1200/"
MERGED_OUTPUT_PATH = "/root/autodl-tmp/merged_model"

def merge_clora_layers(model):
    """
    将模型中的 CLoRALinear 层合并为原始 Linear 层（权重加和）
    """
    for name, module in model.named_modules():
        if isinstance(module, CLoRALinear):
            # 获取合并后的权重：W' = W + A @ B
            merged_weight = module.weight.data + module.A @ module.B

            # 构造新的标准Linear层
            linear = torch.nn.Linear(module.in_features, module.out_features).to(module.weight.device)
            linear.weight.data = merged_weight
            linear.bias = module.bias

            # 替换原模型中的 CLoRALinear 层
            parent = model
            for attr in name.split(".")[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, name.split(".")[-1], linear)

    return model

def main():
    # 加载 base 模型 + CLoRA 层
    print("[INFO] Loading base + CLoRA model...")
    model = AutoModelForCausalLM.from_pretrained(
        CLORA_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    print("[INFO] Merging CLoRA layers into base model...")
    model = merge_clora_layers(model)

    print(f"[INFO] Saving merged model to {MERGED_OUTPUT_PATH}...")
    model.save_pretrained(MERGED_OUTPUT_PATH)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
