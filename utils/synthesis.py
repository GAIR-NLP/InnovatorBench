#!/usr/bin/env python3
"""
数据合成脚本 - 支持caption法和直接COT法
Data synthesis script - supports both caption method and direct CoT method
"""

import argparse
import json
import os
import sys
import time
import random
from tqdm import tqdm
import torch
from PIL import Image
from vllm import LLM, SamplingParams
import re
import gc

def parse_args():
    parser = argparse.ArgumentParser(description='Data synthesis for puzzle-solving model')
    
    # 数据合成参数
    parser.add_argument('--method', type=str, choices=['caption', 'direct'], default='direct',
                        help='数据合成方法: caption (两步法) 或 direct (直接法)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='生成的样本数量')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='生成温度')
    parser.add_argument('--filter-all-correct', action='store_true', default=False,
                        help='是否过滤掉全对数据（保留所有非空预测）')
    parser.add_argument('--n-candidates', type=int, default=8,
                        help='每个样本生成的候选数量')
    # parser.add_argument('--max-tokens', type=int, default=8192,
    #                     help='最大生成token数')
    parser.add_argument('--batch-size', type=int, default=1000000,
                        help='推理批次大小')
    
    # 模型路径
    parser.add_argument('--caption-model', type=str, default="/workspace/data/checkpoints/Qwen2.5-VL-72B-Instruct",
                        help='Caption模型路径')
    parser.add_argument('--reasoning-model', type=str, default="/workspace/data/checkpoints/DeepSeek-R1-Distill-Qwen-32B",
                        help='推理模型路径 (caption法)')
    parser.add_argument('--direct-model', type=str, default="/workspace/data/checkpoints/Qwen2.5-VL-72B-Instruct",
                        help='直接推理模型路径 (direct法)')
    
    # 数据路径
    parser.add_argument('--train-data', type=str, default="/workspace/data/datasets/train/train.jsonl",
                        help='训练数据路径')
    parser.add_argument('--train-images', type=str, default="/workspace/data/datasets/train/train_images",
                        help='训练图像目录')
    
    # 输出路径
    parser.add_argument('--output-dir', type=str, default="/workspace/training_data_cache",
                        help='训练数据缓存目录')
    parser.add_argument('--caption-cache-dir', type=str, default="/tmp/caption_cache",
                        help='Caption缓存目录')
    
    # 硬件参数
    parser.add_argument('--gpus', type=int, default=8,
                        help='使用的GPU数量')
    
    return parser.parse_args()

def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def write_jsonl(data, file_path):
    """写入JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def append_jsonl(data_item, file_path):
    """追加写入JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data_item, ensure_ascii=False) + '\n')

def extract_answer_letter(pred_str):
    """从预测文本中提取答案字母"""
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

def load_training_data(args):
    """加载并选择训练数据"""
    print("📚 加载训练数据...")
    all_data = read_jsonl(args.train_data)
    
    if len(all_data) > args.num_samples:
        selected_data = random.sample(all_data, args.num_samples)
    else:
        selected_data = all_data
        print(f"⚠️  训练数据只有 {len(all_data)} 条，少于请求的 {args.num_samples} 条")
    
    print(f"✅ 选择了 {len(selected_data)} 个样本进行处理")
    return selected_data

def generate_captions(args, samples):
    """生成图像描述 (Caption法的第一步)"""
    print("🖼️  开始生成图像描述...")
    
    caption_cache_file = os.path.join(args.caption_cache_dir, "captions.jsonl")
    
    # 检查缓存
    if os.path.exists(caption_cache_file):
        print("📋 发现缓存的caption，加载中...")
        cached_captions = read_jsonl(caption_cache_file)
        caption_dict = {item['question_id']: item['caption'] for item in cached_captions}
        
        missing_captions = [s for s in samples if s['question_id'] not in caption_dict]
        if not missing_captions:
            print("✅ 所有样本的caption都已缓存")
            for sample in samples:
                sample['image_caption'] = caption_dict[sample['question_id']]
            return samples
        
        print(f"🔄 还需要为{len(missing_captions)}个样本生成caption")
        samples_to_process = missing_captions
    else:
        samples_to_process = samples
        caption_dict = {}
    
    # 初始化caption模型
    print("🚀 初始化Caption模型...")
    caption_llm = LLM(
        model=args.caption_model,
        tensor_parallel_size=min(args.gpus, 4),
        max_model_len=4096,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        gpu_memory_utilization=0.8,
        swap_space=60,
        max_num_seqs=20
    )
    
    # 分批处理
    for i in tqdm(range(0, len(samples_to_process), args.batch_size), desc="生成Caption"):
        batch = samples_to_process[i:i+args.batch_size]
        inputs = []
        
        for sample in batch:
            image_path = os.path.join(args.train_images, sample['image_file'])
            image = Image.open(image_path).convert("RGB")
            
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant that provides detailed descriptions of images.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                "Please describe this image in great detail, including all visually important elements, "
                "shapes, patterns, numbers, text, and spatial relationships. Focus on elements that might "
                "be relevant to solving puzzles or logical reasoning tasks."
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            })
        
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=2048,
            n=1
        )
        
        outputs = caption_llm.generate(inputs, sampling_params=sampling_params)
        
        # 保存caption
        for j, output in enumerate(outputs):
            question_id = batch[j]['question_id']
            caption = output.outputs[0].text
            caption_dict[question_id] = caption
            
            append_jsonl({
                'question_id': question_id,
                'caption': caption
            }, caption_cache_file)
    
    # 释放模型
    del caption_llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # 添加caption到样本
    for sample in samples:
        sample['image_caption'] = caption_dict[sample['question_id']]
    
    print("✅ Caption生成完成")
    return samples

def expand_samples_by_swapping_options(samples):
    """将每个样本按选项进行扩展：与正确选项逐一交换以生成变体。

    规则：
    - 保留原题（标记为 _orig）。
    - 将正确答案与其余每个选项交换，生成 N-1 个新题（标记为 _swap_{LETTER}），最终每题生成 N 个样本。
    - 交换后需要同步更新 choices 的顺序与 answer 的字母。
    - 若样本不满足要求（无 choices、answer 无效等），则原样保留该样本。
    """
    expanded = []
    for sample in samples:
        choices = list(sample.get('choices', []))
        answer_letter = str(sample.get('answer', '')).strip()

        if not choices or len(answer_letter) != 1 or not ('A' <= answer_letter <= 'Z'):
            expanded.append(sample)
            continue

        num_options = len(choices)
        letters = [chr(65 + i) for i in range(num_options)]
        if answer_letter not in letters:
            expanded.append(sample)
            continue

        correct_index = ord(answer_letter) - 65

        # 原题
        original_sample = sample.copy()
        original_sample['question_id'] = f"{sample['question_id']}_orig"
        expanded.append(original_sample)

        # 与其它选项逐一交换
        for idx, letter in enumerate(letters):
            if idx == correct_index:
                continue

            swapped_choices = list(choices)
            swapped_choices[correct_index], swapped_choices[idx] = swapped_choices[idx], swapped_choices[correct_index]

            new_sample = sample.copy()
            new_sample['choices'] = swapped_choices
            new_sample['answer'] = letter
            new_sample['question_id'] = f"{sample['question_id']}_swap_{letter}"

            # 避免沿用可能存在的旧 caption
            if 'image_caption' in new_sample:
                del new_sample['image_caption']

            expanded.append(new_sample)

    return expanded

def synthesize_data_caption_method(args, samples):
    """Caption法：基于描述+问题生成CoT"""
    print("🧠 开始Caption法CoT生成...")
    
    # 初始化推理模型
    print("🚀 初始化推理模型...")
    reasoning_llm = LLM(
        model=args.reasoning_model,
        tensor_parallel_size=args.gpus,
        # max_model_len=args.max_tokens,
        max_model_len=8192,
        gpu_memory_utilization=0.8,
        swap_space=60,
        max_num_seqs=20
    )
    
    results = []
    
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Caption法CoT生成"):
        batch = samples[i:i+args.batch_size]
        inputs = []
        metadata = []
        
        for sample in batch:
            question = sample['question']
            choices = sample['choices']
            answer = sample['answer']
            image_caption = sample['image_caption']
            
            option_str = ""
            for idx, choice in enumerate(choices):
                letter = chr(65 + idx)
                option_str += f"{letter}. {choice}\n"
            
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant that excels at logical reasoning and problem solving.<|im_end|>\n"
                "<|im_start|>user\n"
                f"Image description: {image_caption}\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{option_str}\n"
                "Please think step by step and provide detailed reasoning for this problem. "
                "Use chain-of-thought reasoning to analyze the image description and question carefully. "
                "Put your final answer in \\boxed{} format with only the option letter."
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            inputs.append({"prompt": prompt})
            metadata.append(sample)
        
        sampling_params = SamplingParams(
            temperature=args.temperature,
            # max_tokens=args.max_tokens,
            max_model_len=8192,
            n=args.n_candidates,
            stop=["<|im_end|>", "</s>"]
        )
        
        outputs = reasoning_llm.generate(inputs, sampling_params=sampling_params)
        
        for j, output in enumerate(outputs):
            sample = metadata[j]
            for k in range(args.n_candidates):
                generated_text = output.outputs[k].text
                predicted_answer = extract_answer_letter(generated_text)
                
                result = {
                    "question_id": f"{sample['question_id']}_{k}",
                    "question": sample['question'],
                    "choices": sample['choices'],
                    "image_file": sample['image_file'],
                    "image_caption": sample['image_caption'],
                    "response": generated_text,
                    "predicted_answer": predicted_answer,
                    "ground_truth": sample['answer']
                }
                results.append(result)
    
    del reasoning_llm
    gc.collect()
    torch.cuda.empty_cache()
    
    print("✅ Caption法CoT生成完成")
    return results

def synthesize_data_direct_method(args, samples):
    """直接法：基于图像+问题直接生成CoT"""
    print("🧠 开始直接法CoT生成...")
    
    print("🚀 初始化多模态模型...")
    llm = LLM(
        model=args.direct_model,
        tensor_parallel_size=args.gpus,
        # max_model_len=args.max_tokens,
        max_model_len=8192,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        gpu_memory_utilization=0.8,
        swap_space=60,
        max_num_seqs=20
    )
    
    results = []
    
    for i in tqdm(range(0, len(samples), args.batch_size), desc="直接法CoT生成"):
        batch = samples[i:i+args.batch_size]
        inputs = []
        metadata = []
        
        for sample in batch:
            question = sample['question']
            choices = sample['choices']
            answer = sample['answer']
            image_path = os.path.join(args.train_images, sample['image_file'])
            image = Image.open(image_path).convert("RGB")
            
            option_str = ""
            for idx, choice in enumerate(choices):
                letter = chr(65 + idx)
                option_str += f"{letter}. {choice}\n"
            
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant that excels at logical reasoning and problem solving.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}\n\n"
                f"Options:\n{option_str}\n"
                "Please analyze the image carefully and think step by step to solve this problem. "
                "Use chain-of-thought reasoning to examine all visual elements, patterns, relationships, "
                "and logical constraints. Explain your reasoning process clearly before giving your answer. "
                "Put your final answer in \\boxed{} format with only the option letter."
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            })
            metadata.append(sample)
        
        sampling_params = SamplingParams(
            temperature=args.temperature,
            # max_tokens=args.max_tokens,
            max_model_len=8192,
            n=args.n_candidates,
            stop=["<|im_end|>", "</s>"]
        )
        
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        
        for j, output in enumerate(outputs):
            sample = metadata[j]
            for k in range(args.n_candidates):
                generated_text = output.outputs[k].text
                predicted_answer = extract_answer_letter(generated_text)
                
                result = {
                    "question_id": f"{sample['question_id']}_{k}",
                    "question": sample['question'],
                    "choices": sample['choices'],
                    "image_file": sample['image_file'],
                    "response": generated_text,
                    "predicted_answer": predicted_answer,
                    "ground_truth": sample['answer']
                }
                results.append(result)
    
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    print("✅ 直接法CoT生成完成")
    return results

def filter_and_convert_data(args, results):
    """过滤数据并转换为训练格式"""
    print("🔍 过滤和转换数据...")
    
    if not args.filter_all_correct:
        # 只保留预测正确的结果
        filtered = [r for r in results if r['predicted_answer'] == r['ground_truth']]
        print(f"📊 过滤前: {len(results)} 条, 过滤后: {len(filtered)} 条 (只保留正确预测)")
    else:
        # 保留所有非空预测的结果
        filtered = [r for r in results if r['predicted_answer'] and r['predicted_answer'] != ""]
        print(f"📊 过滤前: {len(results)} 条, 过滤后: {len(filtered)} 条 (保留所有非空预测)")
    
    # 转换为LLaMA-Factory格式
    training_data = []
    for result in filtered:
        option_str = ""
        for idx, choice in enumerate(result['choices']):
            letter = chr(65 + idx)
            option_str += f"{letter}. {choice}\n"
        
        training_sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{result['question']}\n\nOptions:\n{option_str}\nPlease think step by step and provide detailed reasoning for this problem. Put your final answer in \\boxed{{}} format with only the option letter."
                },
                {
                    "from": "gpt", 
                    "value": result['response']
                }
            ],
            "images": [result['image_file']]
        }
        training_data.append(training_sample)
    
    print(f"✅ 转换完成，共 {len(training_data)} 条训练样本")
    return training_data

def main():
    args = parse_args()
    
    print("🚀 开始数据合成")
    print(f"📋 配置参数:")
    print(f"   - 合成方法: {args.method}")
    print(f"   - 样本数量: {args.num_samples}")
    print(f"   - 候选数量: {args.n_candidates}")
    print(f"   - 温度: {args.temperature}")
    print(f"   - GPU数量: {args.gpus}")
    print(f"   - 过滤模式: {'所有非空预测' if args.filter_all_correct else '只保留正确预测'}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.caption_cache_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Step 1: 加载数据
    samples = load_training_data(args)

    # Step 2: 将数据按不同的选项进行扩展，得到多个不同的样本
    samples = expand_samples_by_swapping_options(samples)
    
    # Step 3: 数据合成
    if args.method == 'caption':
        # Caption法：先生成描述，再生成推理
        samples_with_captions = generate_captions(args, samples)
        results = synthesize_data_caption_method(args, samples_with_captions)
    else:
        # 直接法：直接生成推理
        results = synthesize_data_direct_method(args, samples)
    
    # Step 4: 过滤和转换数据
    training_data = filter_and_convert_data(args, results)
    
    # Step 5: 保存结果
    raw_file = os.path.join(args.output_dir, f"{args.method}_synthesis_results.jsonl")
    write_jsonl(results, raw_file)
    print(f"📁 原始结果已保存到: {raw_file}")
    
    training_file = os.path.join(args.output_dir, f"{args.method}_training_data.jsonl")
    write_jsonl(training_data, training_file)
    print(f"📁 训练数据已保存到: {training_file}")
    
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print(f"\n🎉 数据合成完成！")
    print(f"⏱️  总耗时: {hours}小时{minutes}分钟")
    print(f"📊 统计信息:")
    print(f"   - 原始样本: {len(samples)}")
    print(f"   - 生成结果: {len(results)}")
    print(f"   - 训练样本: {len(training_data)}")
    print(f"📁 输出文件:")
    print(f"   - 原始结果: {raw_file}")
    print(f"   - 训练数据: {training_file}")

if __name__ == "__main__":
    main()
