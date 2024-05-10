import torch
import time
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GemmaForCausalLM
# from peft import PeftGemmaForCausalLM
import json

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_name ="./merged_peft/final_merged_checkpoint"
adapter_path = "results/final_checkpoint"
# adapter_path = "./dpo_results/final_checkpoint"
# model_path = "merged_peft/final_merged_checkpoint"
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.bfloat16,
    load_in_4bit=False,
    device_map="auto",
)
# model = GemmaForCausalLM.from_pretrained(
#     model_path, torch_dtype=torch.bfloat16, device_map="auto"
# )

print(model.device)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     load_in_4bit=True,
# )

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
print(len(tokenizer))

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# input_text = "Is it possible to snap a picture using the back camera at a high resolution?"
# # inputs = tokenizer.encode(f"Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {input_text} \n\nResponse: ", return_tensors="pt").to(DEV)
# text = f"Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {input_text} \n\nResponse: "

# inputs = tokenizer(text, return_tensors="pt").to(DEV)
# #     input_ids=inputs,
# #     temperature=0.2, 
# #     top_p=0.95, 
# #     top_k=40,
# #     max_new_tokens=500,
# #     repetition_penalty=1.3
# # )
# # outputs = model.generate(**generate_kwargs)
# # print(tokenizer.decode(outputs[0]))

# outputs = model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def inference(input_text):
    start_time = time.time()
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = input_ids["input_ids"].shape[1]
    outputs = model.generate(
        input_ids=input_ids["input_ids"], 
        # max_length=4096,
        max_new_tokens=1024,
        do_sample=False,
        # temperature=0.9,
        # top_p=0.95,
        # top_k=40,
        # repetition_penalty=1.3,
        )
    generated_sequence = outputs[:, input_length:].tolist()

    res = tokenizer.decode(generated_sequence[0],skip_special_tokens=True)
    end_time = time.time()
    return {"output": res, "latency": end_time - start_time}

new_tokens = [
    "<nexa_0>",
    "<nexa_1>",
    "<nexa_2>",
    "<nexa_3>",
    "<nexa_4>",
    "<nexa_5>",
    "<nexa_6>",
    "<nexa_7>",
    "<nexa_8>",
    "<nexa_9>",
    "<nexa_20>",
    "<nexa_end>",
]

def load_text_as_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text.split('\n\n')
fun_examples = load_text_as_list('android_functions.txt')

cls_list = new_tokens[:10]
# print(len(cls_list))
# print(cls_list)
# import pdb
# pdb.set_trace()
class_examples = {
    cls_list[i]: fun_examples[i] for i in range(len(cls_list))
}
# print(class_examples)

function_description = json.dumps(class_examples)
print(function_description)

input_text = "Take a selfie for me with front camera"
# nexa_query = f"""Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {input_text} \n\nResponse: """
nexa_query = f"""
Below is the query from the users, please choose the correct function and generate the
parameters to call the function. 
Function list: {cls_list}, 
Function description: {function_description}
Query: {input_text}
# for single function call
Response:
"""
start_time = time.time()
print("nexa model result:\n", inference(nexa_query))
print("latency:", time.time() - start_time," s")