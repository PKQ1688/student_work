import os
import json
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from torch import nn
from utils import find_all_linear_names, print_trainable_parameters

debug_model = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(True)

output_dir = "results"
model_name = "gemma-2b"

def load_text_as_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text.split('\n\n')
fun_examples = load_text_as_list('android_functions.txt')


print(len(fun_examples))
# print(class_examples)
# exit()

# 定义你想添加的新词汇
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
# exit()
# exit()

dataset = load_dataset("json", data_files="data/*.json",split="train")
# dataset = load_dataset("json", data_files="data/nexa_0.json", split="train")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config,low_cpu_mem_usage=True
)
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
# )
# base_model.to(device)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
print(tokenizer.default_chat_template)
# exit()

if debug_model:
    text = "Quote: Imagination is more"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = base_model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))



# 添加新词汇到词汇表
num_added_toks = tokenizer.add_tokens(new_tokens)
print("We have added", num_added_toks, "tokens")
print(len(tokenizer))

# 为新tokens扩充模型的embedding层
base_model.resize_token_embeddings(len(tokenizer))

# Change the LORA hyperparameters accordingly to fit your use case
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    # target_modules=find_all_linear_names(base_model),
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# peft_config = LoraConfig(
#     r=8,
#     target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
#     task_type="CAUSAL_LM",
# )

# model = get_peft_model(base_model, peft_config)
model = base_model
print_trainable_parameters(model)
# model.train()

# for param in model.parameters():
# param.requires_grad = True  # 确保所有参数都需要梯度
# n_candidates = len(new_tokens) - 1
def formatting_prompts_func(example):
    output_texts = []
    # import pdb
    # pdb.set_trace()
    for i in range(len(example["input"])):
        # text = f"### Input: ```{example['input'][i]}```\n ### Output: {example['output'][i]}"
        # text = f"Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {example['input'][i]} \n\nResponse: {example['output'][i]}"
        text = f"""
Below is the query from the users, please choose the correct function and generate the
parameters to call the function. 
Function list: {cls_list}, 
Function description: {function_description}
Query: {example['input'][i]}
# for single function call
Response: {example['output'][i].split("Function description:")[0]}
"""
        # import pdb
        # pdb.set_trace()
        output_texts.append(text)
    # import pdb
    # pdb.set_trace()
    return output_texts


# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    num_train_epochs=1,
    learning_rate=5e-5,
    bf16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir=output_dir,
    # optim="paged_adamw_32bit",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)
# args=transformers.TrainingArguments(
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     warmup_steps=2,
#     max_steps=10,
#     learning_rate=2e-4,
#     fp16=True,
#     logging_steps=1,
#     output_dir="outputs",
#     optim="paged_adamw_8bit"
# ),
class MySFTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写 compute_loss 方法来自定义损失计算。
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        input_ids = inputs['input_ids']
        labels = input_ids.clone()  # 假设标签是输入的一个复制，通常标签会稍有不同

        # 前向传播
        outputs = model(**inputs, labels=labels)

        # 获取logits
        logits = outputs.logits  # 正确的属性
        # import pdb
        # pdb.set_trace()
        # 分类任务
        # classification_logits = self.classification_head(sequence_output[:, 0, :])  # 取第一个Token的隐藏状态
        # classification_logits = self.softmax(classification_logits)  # 应用Softmax

        loss_fct = nn.CrossEntropyLoss()
        first_token_loss = loss_fct(logits[:, 0, :], labels[:, 0])
        # first_token_loss = loss_fct(logits[:, 0, :], labels[:, 0].unsqueeze(0))  # 第一个token使用softmax
        subsequent_tokens_loss = loss_fct(logits[:, 1:, :].transpose(1, 2), labels[:, 1:])  # 后续tokens使用交叉熵

        total_loss = first_token_loss + subsequent_tokens_loss

        return (total_loss, outputs) if return_outputs else total_loss

trainer = MySFTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=1024,
    formatting_func=formatting_prompts_func,
    args=training_args,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
