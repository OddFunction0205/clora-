import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from clora.modeling_clora import CLoRALinear
from clora.utils import generate_orthogonal_matrix
from transformers import Trainer
from torch.optim import AdamW
from transformers import get_scheduler

import os
print("TORCH_USE_DTENSOR:", os.environ.get("TORCH_USE_DTENSOR"))


# ========== 环境与路径配置 ==========
HF_DATASETS_CACHE = "/root/autodl-tmp/data/commonsense_170k"
TRANSFORMERS_CACHE = "/root/autodl-tmp/model"
MODEL_CACHE = "/root/autodl-tmp/model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# ========== 数据预处理 ==========
def preprocess(example):
    text = example['instruction'] + ("\n" + example['input'] if example['input'] else "") + "\n" + str(example['output'])
    return {"text": text}

print("[INFO] Loading Commonsense170K dataset")
dataset = load_from_disk(HF_DATASETS_CACHE)
dataset = dataset.map(preprocess)

print("[INFO] Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=TRANSFORMERS_CACHE,local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

if "validation" in dataset:
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
else:
    split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

# ========== 注入CLoRA层 ==========
def inject_clora(model, r, k, lambda_orth):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ("q_proj" in name or "v_proj" in name):
            device = module.weight.device
            dtype = module.weight.dtype
            clora = CLoRALinear(
                module.in_features, module.out_features,
                r=r, k=k, lambda_orth=lambda_orth,
                device=device
            ).to(device).to(dtype)
            
            clora.P_A.data.copy_(
                generate_orthogonal_matrix(module.out_features, k, device=device, dtype=dtype)
            )
            clora.P_B.data.copy_(
                generate_orthogonal_matrix(module.in_features, k, device=device, dtype=dtype)
            )

            # 打印调试信息
            print(f"[DEBUG] Injected at {name}, dtype={dtype}, device={device}")

            parent = model
            for attr in name.split(".")[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, name.split(".")[-1], clora)
    print(f"[DEBUG] Injected {name}:")
    print(f"  A.dtype: {clora.A.dtype}, B.dtype: {clora.B.dtype}")
    print(f"  P_A.dtype: {clora.P_A.dtype}, P_B.dtype: {clora.P_B.dtype}")
    print(f"  Device: {clora.A.device}, Model Weight Device: {module.weight.device}")



print("[INFO] Loading model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CACHE,
    torch_dtype=torch.float16,
    cache_dir=TRANSFORMERS_CACHE,
    local_files_only=True
)
model.to("cuda:0")

print(f"模型权重所在设备: {next(model.parameters()).device}")
inject_clora(model, r=32, k=512, lambda_orth=1.0)
print(f"注入CLoRA后模型权重设备: {next(model.parameters()).device}")

# ========== 自定义Trainer ==========
class CLoRATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        orth_loss = 0.0
        for module in model.modules():
            if isinstance(module, CLoRALinear):
                orth_loss += module.orthogonal_loss()
        total_loss = loss + orth_loss
        return (total_loss, outputs) if return_outputs else total_loss

# ========== 设置训练参数 ==========
training_args = TrainingArguments(
    output_dir="/root/autodl-tmp/clora_commonsense_model",
    num_train_epochs=3,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=8,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=3e-4,
    weight_decay=0.01,
    fp16=False,
    optim="adamw_torch",
    report_to=[],
)

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

trainer = CLoRATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    optimizers=(optimizer, None)
)

# ========== 开始训练 ==========
trainer.train()

# ========== 保存模型与评估 ==========
trainer.save_model("./clora_commonsense_final")
eval_metrics = trainer.evaluate()
print("[INFO] Eval Metrics:", eval_metrics)
