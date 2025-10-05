import os
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import glob
from datetime import datetime


@dataclass
class EvalInfo:
    timestamp: str
    action_type: str
    duration: float = 0.0
    workspace: str = ""
    file_name: str = ""
    score: int = 0

def extract_eval_times(nodes_dir: str, workspace_name: str) -> list[EvalInfo]:
    """从 nodes_dir 中提取所有 eval 动作的运行时间"""
    eval_infos = []
    
    # 查找所有 JSON 文件
    json_files = glob.glob(os.path.join(nodes_dir, "*.json"))
    if not json_files:
        return eval_infos
    
    # 首先找到 root 的 timestamp
    root_timestamp = None
    root_file = None
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 查找 root 节点
            if "root" in json_file:
                root_timestamp = data.get("timestamp")
                root_file = json_file
                break
        except Exception as e:
            print(f"警告: 无法解析文件 {json_file}: {e}")
    
    if not root_timestamp:
        print(f"警告: 在 {nodes_dir} 中未找到 root 节点")
        return eval_infos
    
    # 转换 root timestamp 为 datetime
    try:
        root_time = datetime.fromisoformat(root_timestamp.replace('Z', '+00:00'))
    except Exception as e:
        print(f"警告: 无法解析 root timestamp: {e}")
        return eval_infos
    
    # 查找所有 eval 动作
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否有 action 字段
            if isinstance(data, dict) and "action" in data:
                action = data["action"]
                obs = data["observation"]
                if isinstance(action, dict) and action.get("action_type") == "eval":
                    timestamp = data.get("timestamp")
                    if timestamp:
                        try:
                            eval_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            duration = (eval_time - root_time).total_seconds()
                            
                            eval_info = EvalInfo(
                                timestamp=timestamp,
                                action_type="eval",
                                duration=duration,
                                workspace=workspace_name,
                                file_name=json_file,
                                score=obs.get("overall_score", 0)
                            )
                            eval_infos.append(eval_info)
                        except Exception as e:
                            print(f"警告: 无法解析 timestamp {timestamp}: {e}")
        
        except Exception as e:
            print(f"警告: 无法处理文件 {json_file}: {e}")
    
    # 按 timestamp 排序
    eval_infos.sort(key=lambda x: x.timestamp)
    return eval_infos


def main(argv: Optional[Iterable[str]] = None) -> int:
    benchmark_file = "benchmark.txt"
    if not os.path.exists(benchmark_file):
        print(f"错误: 文件 '{benchmark_file}' 不存在")
        return 1
    
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        workspace_dirs = [line.strip() for line in f if line.strip()]
    
    if not workspace_dirs:
        print("benchmark.txt 中没有有效的 workspace_dir")
        return 1

    all_eval_infos = []
    
    for directory in workspace_dirs:
        # 构建绝对路径
        abs_dir = "/inspire/hdd/project/qproject-fundationmodel/liupengfei-24025/ai_engineer/workspace_backup"
        nodes_dir = os.path.join(abs_dir, directory, "nodes")
        
        if not os.path.exists(nodes_dir):
            print(f"警告: 目录 {nodes_dir} 不存在")
            continue
        
        print(f"处理目录: {nodes_dir}")
        eval_infos = extract_eval_times(nodes_dir, directory)
        all_eval_infos.extend(eval_infos)
        
        # 打印当前目录的结果
        if eval_infos:
            print(f"  找到 {len(eval_infos)} 个 eval 动作")
            for info in eval_infos:
                print(f"    运行时间: {info.duration:.2f} 秒, 文件名：{info.file_name}")
        else:
            print("  未找到 eval 动作")
    
    # 输出汇总结果
    if all_eval_infos:
        print("\n=== 汇总结果 ===")
        total_evals = len(all_eval_infos)
        total_duration = sum(info.duration for info in all_eval_infos)
        avg_duration = total_duration / total_evals
        
        print(f"总共找到 {total_evals} 个 eval 动作")
        print(f"总运行时间: {total_duration:.2f} 秒")
        print(f"平均运行时间: {avg_duration:.2f} 秒")
        
        # 按 workspace 分组统计
        workspace_stats = {}
        eval_times_stats = {}
        score_stats = {}
        for info in all_eval_infos:
            if info.workspace not in workspace_stats:
                workspace_stats[info.workspace] = []
                eval_times_stats[info.workspace] = []
                score_stats[info.workspace] = []
            workspace_stats[info.workspace].append(info.duration)
            eval_times_stats[info.workspace].append(info.duration)
            score_stats[info.workspace].append(info.score)
        
        print("\n按 workspace 统计:")
        for workspace, durations in workspace_stats.items():
            count = len(durations)
            total = sum(durations)
            avg = total / count
            print(f"  {workspace}: {count} 个 eval, 总时间 {total:.2f}s, 平均 {avg:.2f}s")
        
        # 保存详细结果到文件
        output_file = "eval_times_report.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("eval 动作运行时间报告\n")
            # f.write("=" * 50 + "\n\n")
            
            # 竖排打印：对齐各 workspace 的第 1、2、3... 次 eval
            workspaces = list(eval_times_stats.keys())
            duration_rows = [eval_times_stats[w] for w in workspaces]
            max_len = max((len(row) for row in duration_rows), default=0)

            for col_idx in range(max_len):
                for row in duration_rows:
                    if col_idx < len(row):
                        f.write(f"{row[col_idx]:.2f}\n")
                    else:
                        f.write("\n")
                if col_idx != max_len - 1:
                    f.write("----------\n")

            f.write(f"\n总计: {total_evals} 个 eval 动作\n")
            f.write(f"总运行时间: {total_duration:.2f} 秒\n")
            f.write(f"平均运行时间: {avg_duration:.2f} 秒\n")
            
        score_output_file = "eval_score_report.txt"
        with open(score_output_file, 'w', encoding='utf-8') as f:
            f.write("eval 动作得分报告\n")
            # 竖排打印：对齐各 workspace 的第 1、2、3... 次得分
            workspaces = list(score_stats.keys())
            score_rows = [score_stats[w] for w in workspaces]
            max_len = max((len(row) for row in score_rows), default=0)

            for col_idx in range(max_len):
                for row in score_rows:
                    if col_idx < len(row):
                        f.write(f"{row[col_idx]:.2f}\n")
                    else:
                        f.write("\n")
                if col_idx != max_len - 1:
                    f.write("----------\n")

        print(f"\n详细结果已保存到: {output_file}")
    else:
        print("未找到任何 eval 动作")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())