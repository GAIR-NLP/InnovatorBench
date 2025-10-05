#!/usr/bin/env python3
"""
统计nodes文件夹中所有node文件的时间跨度
文件格式: {node_type}_{node_id}_{datetime}.json
例如: react_1c01b226-23ae-417a-9690-341d4db79499_2025-09-07T14:19:10.593530.json
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional


def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    """
    从文件名中提取datetime
    格式: {node_type}_{node_id}_{datetime}.json
    """
    try:
        # 移除.json后缀
        if not filename.endswith('.json'):
            return None
        
        base_name = filename[:-5]  # 移除.json
        
        # 分割文件名，获取最后一部分作为时间戳
        parts = base_name.split('_')
        if len(parts) < 3:
            return None
            
        datetime_str = parts[-1]
        
        # 解析ISO格式的时间戳
        return datetime.fromisoformat(datetime_str)
        
    except (ValueError, IndexError) as e:
        print(f"警告: 无法解析文件名 {filename}: {e}")
        return None


def find_all_node_files(nodes_dir: str = "nodes") -> List[Tuple[str, datetime]]:
    """
    查找nodes文件夹中的所有node文件并提取时间
    返回: [(filename, datetime), ...]
    """
    if not os.path.exists(nodes_dir):
        print(f"错误: 文件夹 '{nodes_dir}' 不存在")
        return []
    
    node_files = []
    
    for filename in os.listdir(nodes_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(nodes_dir, filename)
            if os.path.isfile(file_path):
                dt = extract_datetime_from_filename(filename)
                if dt:
                    node_files.append((filename, dt))
    
    return node_files


def calculate_duration(node_files: List[Tuple[str, datetime]]) -> Optional[Tuple[datetime, datetime, float]]:
    """
    计算时间跨度
    返回: (开始时间, 结束时间, 持续时间(秒))
    """
    if not node_files:
        return None
    
    # 按时间排序
    sorted_files = sorted(node_files, key=lambda x: x[1])
    
    start_time = sorted_files[0][1]
    end_time = sorted_files[-1][1]
    
    duration_seconds = (end_time - start_time).total_seconds()
    
    return start_time, end_time, duration_seconds


def format_duration(seconds: float) -> str:
    """格式化持续时间显示为 x小时y分钟z秒"""
    hours = int(seconds // 3600)
    remaining = seconds % 3600
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0:
        parts.append(f"{minutes}分钟")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}秒")
    
    return "".join(parts)


def main():
    """主函数"""
    
    benchmark_file = "benchmark.txt"
    if not os.path.exists(benchmark_file):
        print(f"错误: 文件 '{benchmark_file}' 不存在")
        return
    
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        workspace_dirs = [line.strip() for line in f if line.strip()]
    
    if not workspace_dirs:
        print("benchmark.txt 中没有有效的 workspace_dir")
        return
    
    for workspace_dir in workspace_dirs:

        abs_dir = "/inspire/hdd/project/qproject-fundationmodel/liupengfei-24025/ai_engineer/workspace_backup"
        nodes_dir = os.path.join(abs_dir, workspace_dir, "nodes")
        
        print(f"正在扫描文件夹: {nodes_dir}")
        node_files = find_all_node_files(nodes_dir)
        
        if not node_files:
            print("未找到有效的node文件")
            return
        
        print(f"找到 {len(node_files)} 个node文件")
        
        # 计算时间跨度
        result = calculate_duration(node_files)
        if not result:
            print("无法计算时间跨度")
            return
        
        start_time, end_time, duration_seconds = result
        
        print("\n=== 时间统计结果 ===")
        print(f"开始时间: {start_time}")
        print(f"结束时间: {end_time}")
        print(f"总运行时间: {format_duration(duration_seconds)}")
        print(f"原始秒数: {duration_seconds:.3f}秒")
        
        # 显示前5个和后5个文件
        sorted_files = sorted(node_files, key=lambda x: x[1])
        print(f"\n=== 文件列表 (共{len(sorted_files)}个) ===")
        
        display_limit = 5
        if len(sorted_files) <= display_limit * 2:
            for filename, dt in sorted_files:
                print(f"{dt}: {filename}")
        else:
            print("最早的5个文件:")
            for filename, dt in sorted_files[:display_limit]:
                print(f"{dt}: {filename}")
            print("...")
            print("最晚的5个文件:")
            for filename, dt in sorted_files[-display_limit:]:
                print(f"{dt}: {filename}")

        # 将总运行时间和原始秒数分别写入日志文件
        with open("time_duration.log", "a", encoding="utf-8") as f:
            f.write(f"总运行时间: {format_duration(duration_seconds)}\n")
        
        with open("origin_time_duration.log", "a", encoding="utf-8") as f:
            f.write(f"原始秒数: {duration_seconds:.3f}秒\n")

if __name__ == "__main__":
    main()
