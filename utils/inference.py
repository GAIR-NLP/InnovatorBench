import argparse
import json
import os
from tqdm import tqdm
import torch
from PIL import Image
import random
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
import re
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with a vision-language model on a dataset')
    parser.add_argument('--model_path', type=str, default="",
                        help='Path to the model')
    parser.add_argument('--input_file', type=str, default="",
                        help='Path to input JSONL file')
    parser.add_argument('--image_folder', type=str, default="",
                        help='Path to folder containing images')
    parser.add_argument('--output_file', type=str, default="",
                        help='Path to output JSONL file')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--majority_voting', action='store_true',
                        help='Enable majority voting across multiple samples')
    parser.add_argument('--num_votes', type=int, default=5,
                        help='Number of samples to generate for majority voting')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature when majority voting is enabled')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    
    return parser.parse_args()


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_answer_letter(pred_str):
    pred_str = pred_str.replace("\u043a\u0438", "")
    pred = ""
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    
    if pred and pred not in [chr(65+i) for i in range(26)]:
        pred = random.choices(["A", "B", "C", "D", "E"])[0]
    elif not pred:
        pred = random.choices(["A", "B", "C", "D", "E"])[0]
        
    return pred


def majority_vote(answers):
    counter = Counter(answers)
    if not counter:
        return random.choices(["A", "B", "C", "D", "E"])[0]
    most_common = counter.most_common()
    max_count = most_common[0][1]
    tied_answers = sorted([ans for ans, cnt in most_common if cnt == max_count])
    return tied_answers[0]


def run_inference(model_path, input_data, image_folder, num_gpus, majority_voting=False, num_votes=1, temperature=0.0, seed=None):
    # Initialize the vision-language model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        max_model_len=4096,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        gpu_memory_utilization=0.96,
        swap_space=60,
        max_num_seqs=32
    )
    
    results = []
    
    # Process all data at once for batch inference
    batch = input_data
        
    inputs = []
    for item in batch:
        question_id = item["question_id"]
        question = item["question"]
        choices = item["choices"]
        image_file = item["image_file"]
        
        # Prepare the image
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        
        option_str = ""
        for idx, choice in enumerate(choices):
            letter = chr(65 + idx)
            option_str += f"{letter}: {choice}\n"

        # Format prompt for Qwen2-VL
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}\n"
            f"{option_str}"
            "Please reason step by step and put your final answer with \\boxed{} where only the option letter should be put in the box. For example, \\boxed{A}, \\boxed{B}, \\boxed{C}, etc."
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": image,
            }
        })
    
    # Set sampling parameters
    if majority_voting:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=2048,
            n=num_votes,
            seed=seed,
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            n=1,
            seed=seed,
        )
    
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    # Process outputs
    for i, output in enumerate(outputs):
        # Match output with input by index
        question_id = batch[i]["question_id"]
        
        if majority_voting:
            candidate_answers = [extract_answer_letter(candidate.text) for candidate in output.outputs]
            answer_letter = majority_vote(candidate_answers)
        else:
            generated_text = output.outputs[0].text
            # Extract answer letter from the generated text
            answer_letter = extract_answer_letter(generated_text)
        
        results.append({
            "question_id": question_id,
            "answer": answer_letter
        })
    
    return results

def main():
    args = parse_args()
    
    print(f"Loading data from {args.input_file}")
    input_data = read_jsonl(args.input_file)
    print(f"Loaded {len(input_data)} examples")
    print(f"Running inference with model: {args.model_path}")
    
    results = run_inference(
        model_path=args.model_path,
        input_data=input_data,
        image_folder=args.image_folder,
        num_gpus=args.gpus,
        majority_voting=args.majority_voting,
        num_votes=args.num_votes,
        temperature=args.temperature,
        seed=args.seed,
    )
    
    print(f"Writing results to {args.output_file}")
    write_jsonl(results, args.output_file)

if __name__ == "__main__":
    main()