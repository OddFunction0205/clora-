import os
import torch
import logging
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from clora.modeling_clora import CLoRALinear
from clora.utils import generate_orthogonal_matrix
from transformers import Trainer
from torch.optim import AdamW
from torch.cuda.amp import autocast

# ========== logging 配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("train.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

logger.info(f"TORCH_USE_DTENSOR: {os.environ.get('TORCH_USE_DTENSOR')}")

# ========== 环境与路径配置 ==========
HF_DATASETS_CACHE = "/root/autodl-tmp/data/commonsense_170k"
TRANSFORMERS_CACHE = "/root/autodl-tmp/model"
MODEL_CACHE = "/root/autodl-tmp/model/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# ========== 数据预处理 ==========
def preprocess(example):
    text = example['instruction'] + ("\n" + example['input'] if example['input'] else "") + "\n" + str(example['output'])
    return {"text": text}

logger.info("[INFO] Loading Commonsense170K dataset")
dataset = load_from_disk(HF_DATASETS_CACHE)
dataset = dataset.map(preprocess)

logger.info("[INFO] Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, cache_dir=TRANSFORMERS_CACHE, local_files_only=True)
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
                module.in_features,
                module.out_features,
                r=r, k=k, lambda_orth=lambda_orth,
                device=device,
                bias=module.bias is not None
            ).to(device).to(dtype)

            # 拷贝原始权重和 bias
            clora.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                clora.bias.data.copy_(module.bias.data)

            # 初始化正交矩阵
            clora.P_A.data.copy_(
                generate_orthogonal_matrix(module.out_features, k, device=device, dtype=dtype) * 0.01
            )
            clora.P_B.data.copy_(
                generate_orthogonal_matrix(module.in_features, k, device=device, dtype=dtype) * 0.01
            )

            clora.P_A.requires_grad = False
            clora.P_B.requires_grad = False

            # 替换原始模块
            parent = model
            for attr in name.split(".")[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, name.split(".")[-1], clora)

            logger.debug(f"[DEBUG] Injected CLoRA at {name}")


logger.info("[INFO] Loading model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CACHE,
    torch_dtype=torch.bfloat16,
    cache_dir=TRANSFORMERS_CACHE,
    local_files_only=True
)
model.to("cuda:0")

logger.info(f"模型权重所在设备: {next(model.parameters()).device}")
inject_clora(model, r=32, k=128, lambda_orth=1.0)
logger.info(f"注入CLoRA后模型权重设备: {next(model.parameters()).device}")

# ========== 自定义Trainer ==========
class CLoRATrainer(Trainer):
    def training_step(self, model, inputs, *args, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with autocast(dtype=torch.bfloat16):
            outputs = model(**inputs)
            loss = outputs.loss
            orth_loss = 0.0
            for module in model.modules():
                if isinstance(module, CLoRALinear):
                    orth_loss += module.orthogonal_loss()
            total_loss = loss + orth_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

        logger.info(f"[INFO] loss: {loss.item():.4f}, orth_loss: {orth_loss.item():.4f}, total_loss: {total_loss.item():.4f}, grad_norm: {grad_norm:.4f}")

        self.optimizer.step()

        return total_loss.detach()



# ========== 设置训练参数 ==========
training_args = TrainingArguments(
    output_dir="/root/autodl-tmp/clora_commonsense_model",
    num_train_epochs=3,
    per_device_train_batch_size=48,
    gradient_accumulation_steps=8,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=5,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    optim="adamw_torch",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to=[],
)

optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

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
trainer.save_model("/root/autodl-tmp/clora_commonsense_model")
eval_metrics = trainer.evaluate()
logger.info(f"[INFO] Eval Metrics: {eval_metrics}")
