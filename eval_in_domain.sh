# 运行测试
lm_eval \
  --model hf \
  --model_args pretrained=/root/autodl-tmp/model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9,trust_remote_code=True \
  --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa \
  --device cuda:0 \
  --batch_size 1 \
  --use_cache /root/autodl-tmp/data \
  --output_path results_in_domain.json
