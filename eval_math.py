from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./clora_commonsense_final", torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

input_text = "Question: If Alice has 3 apples and gives Bob 1, how many does she have left? Let's think step by step."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()

outputs = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
