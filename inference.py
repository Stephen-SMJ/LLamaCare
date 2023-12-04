import fire
import json
import jsonlines
import os  # Operating system functionalities
import sys
import torch
from peft import LoraConfig, get_peft_model  # Packages for parameter-efficient fine-tuning (PEFT)
from torch import nn
from tqdm import tqdm
from transformers import GenerationConfig
from transformers import (
    # AutoModel for language modeling tasks
    # AutoTokenizer for tokenization
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,  # Configuration for BitsAndBytes
    # Argument parser for Hugging Face models
    # Training arguments for model training
    # Creating pipelines for model inference
    # Logging information during training
)

from param_dict import *

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def evaluate(
        batch_data,
        tokenizer,
        model,
        input=None,
        temperature=1,
        top_p=0.9,
        top_k=40,
        num_beams=1,
        max_new_tokens=1024,
        **kwargs,
):
    prompts = generate_prompt(batch_data, input)
    inputs = tokenizer(prompts, return_tensors="pt", max_length=max_seq_length, truncation=True, padding=False)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        output_hidden_states=True,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
        print(model.device)
        inputs = tokenizer(prompts, return_tensors="pt", max_length=max_seq_length, truncation=True, padding='max_length')["input_ids"].to("cuda:0")
        print(inputs.device)
        outputs = model(input_ids=inputs)
        last_hidden_state = outputs.hidden_states[-1][-1]
        embedding_flatten = last_hidden_state.view(1,last_hidden_state.shape[0] * last_hidden_state.shape[1])  # [2,131072]
        embedding_flatten = embedding_flatten.to(model.device)
        classification_output = model.classification_head(embedding_flatten)
        label = torch.argmax(classification_output, dim=1)
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)

    return output, label


def generate_prompt(instruction, input=None):
    Prompt = (
        """ <<SYS>>\nYou are a helpful, respectful and honest assistant. 
        Always answer as helpfully as possible, while being safe.  
        Your answers should not include any harmful, unethical, 
        racist, sexist, toxic, dangerous, or illegal content. 
        Please ensure that your responses are socially unbiased and positive in nature.\n\n
        If a question does not make any sense, or is not factually coherent, 
        explain why instead of answering something not correct. 
        If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n[INST] {query} [/INST]"""
    ).format(query=instruction)
    return Prompt
    # return (
    #     {instruction}### Generation Output:"
    # ).format(instruction=instruction)


def main(
        load_8bit: bool = True,
        base_model: str = model_name,
        input_data_path = dataset_name,
        # output_data_path = "Output.jsonl",
):
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config, # 如果需要省内存，可以打开这个
        device_map=device_map,
    )
    print("base model on devcie:{}".format(model.device))
    

    # 加载分类头的权重，如果有的话
    
    if os.path.exists(classification_head_path):
        classification_head = torch.load(classification_head_path)
        print(classification_head.weight)
        model.classification_head = classification_head
        model.classification_head.to(model.device)
        print("load classfication head in {}".format(classification_head_path))
        print(model)
        
    else:
        model.classification_head = nn.Linear(max_seq_length * 5120, 3, bias=False).to(model.device)
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.update({"output_hidden_states": True})

    # Step 5 :Load LLaMA tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # print(model)
    # Step 6 :Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # model = get_peft_model(model, peft_config)
    # model.config.pad_token_id = tokenizer.pad_token_id

    # if not load_8bit:
    #     model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # input_data = jsonlines.open(input_data_path, mode='r')
    input_data = json.load(open(input_data_path, 'r'))
    output_data_path = os.path.join(output_dir,"eval.jsonl")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_data = jsonlines.open(output_data_path, mode='w')
    # pred_sqls = open("/opt/data/private/research/codellama/Llama-X/data/pred_wizardcoder.sql", 'w')
    lable_map = {0: "no", 1: "yes", 2: "maybe"} # 具体
    for id, one_data in tqdm(enumerate(input_data), total=len(input_data), desc="Predicting: "):
        instruction = one_data["prompt_question"]
        # database_info = one_data["input"]

        _output, lable = evaluate(instruction, tokenizer, model)
        try:
            final_output = _output[0].split("[/INST]")[1].strip()
            # final_output = _output[0]
            new_data = {
                "id": id,
                "output": final_output,
                "lable": lable_map[lable.item()]
            }
            output_data.write(new_data)
        except:
            new_data = {
                "id": id,
                "output": "error",
                "lable": lable_map[lable.item()]
            }
            output_data.write(new_data)
            print("error")
    #     pred_sqls.write(final_output + "\n")
    #     # print(instruction,"-->", final_output)
    # pred_sqls.close()


if __name__ == "__main__":
    # fire.Fire(main)
    
    
