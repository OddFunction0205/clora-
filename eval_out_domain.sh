# 运行 out-domain 任务
lm_eval \
  --model hf \
  --model_args pretrained=./clora_commonsense_final,trust_remote_code=True \
  --tasks bigbenchhard,mmlu_pro \
  --device cuda:0 \
  --batch_size 4 \
  --use_cache /root/autodl-tmp/data \
  --output_path results_out_domain.json
