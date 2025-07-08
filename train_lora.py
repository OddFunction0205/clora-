

def main():
    # 所有原有代码内容都移进来
    import os
    import torch
    import logging
    from datasets import load_from_disk
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from transformers import DataCollatorForLanguageModeling
    from torch.optim import AdamW
    from peft import get_peft_model, LoraConfig, TaskType
    from torch.cuda.amp import autocast

    # ========== logging 配置 ==========
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler("train_lora.log", mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    logger.info(f"TORCH_USE_DTENSOR: {os.environ.get('TORCH_USE_DTENSOR')}")

    # ========== 环境与路径配置 ==========
    HF_DATASETS_CACHE = "/root/autodl-tmp/data/commonsense_170k"
    TRANSFORMERS_CACHE = "/root/autodl-tmp/model"
    MODEL_CACHE = "/root/autodl-tmp/model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"

    # ========== 数据预处理 ==========
    def preprocess(example):
        text = example['instruction'] + ("\n" + example['input'] if example['input'] else "") + "\n" + str(example['output'])
        return {"text": text}

    logger.info("[INFO] Loading Commonsense170K dataset")
    dataset = load_from_disk(HF_DATASETS_CACHE)
    dataset = dataset.map(preprocess)

    logger.info("[INFO] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=TRANSFORMERS_CACHE, local_files_only=True)
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

    # ========== 加载基础模型 ==========
    logger.info("[INFO] Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CACHE,
        torch_dtype=torch.bfloat16,
        cache_dir=TRANSFORMERS_CACHE,
        local_files_only=True
    )
    model.to("cuda:0")


    logger.info(f"模型权重所在设备: {next(model.parameters()).device}")

    # ========== LoRA配置并注入 ==========
    lora_config = LoraConfig(
        r=16,                       # LoRA秩，常用8-32
        lora_alpha=32,              # LoRA放大系数
        target_modules=["q_proj", "v_proj"],  # 针对Transformer中q_proj和v_proj层注入LoRA
        lora_dropout=0.1,           # dropout概率
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    logger.info("[INFO] LoRA modules injected")

    # ========== 自定义Trainer，打印训练loss ==========
    class LoRATrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            with autocast(dtype=torch.bfloat16):
                outputs = model(**inputs)
                loss = outputs.loss
            logger.debug(f"[DEBUG] loss: {loss.item():.4f}")
            return (loss, outputs) if return_outputs else loss


    # ========== 训练参数 ==========
    training_args = TrainingArguments(
        output_dir="/root/autodl-tmp/lora_commonsense_model",
        num_train_epochs=3,
        per_device_train_batch_size=24,
        gradient_accumulation_steps=4,
        save_steps=500,
        logging_dir="./logs",
        logging_steps=5,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        dataloader_num_workers=4,
        max_grad_norm=0.3,
        optim="adamw_torch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to=[],
    )

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    trainer = LoRATrainer(
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
    trainer.save_model("/root/autodl-tmp/lora_commonsense_model")
    eval_metrics = trainer.evaluate()
    logger.info(f"[INFO] Eval Metrics: {eval_metrics}")

if __name__ == "__main__":
    main()
