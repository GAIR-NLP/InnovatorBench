#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细检查项目中Python文件中的中文字符，并显示具体内容
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def find_chinese_in_text(text, file_path, line_number=0):
    """
    在文本中查找中文字符，并返回详细的匹配信息
    """
    chinese_pattern = re.compile(r'([\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff])')

    matches = []
    lines = text.split('\n')

    for i, line in enumerate(lines, 1):
        chinese_chars = chinese_pattern.findall(line)
        if chinese_chars:
            # 过滤掉纯标点符号的行
            punctuation_only = re.match(r'^[\s\u3000-\u303f]*$', line)
            if not punctuation_only:
                matches.append({
                    'line_number': line_number + i,
                    'line_content': line.strip(),
                    'chinese_chars': chinese_chars,
                    'file_path': file_path
                })

    return matches

def check_python_files_detailed(root_dir, max_files=10):
    """
    详细检查目录下的所有Python文件中的中文字符
    """
    python_files = []

    # 递归查找所有.py文件
    for py_file in Path(root_dir).rglob('*.py'):
        if py_file.is_file() and not any(part.startswith('.') for part in py_file.parts):
            python_files.append(py_file)

    files_with_chinese = []
    total_chinese_files = 0

    print(f"找到 {len(python_files)} 个Python文件，开始详细检查中文字符...")

    for i, py_file in enumerate(python_files[:max_files]):  # 只检查前max_files个文件作为示例
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            matches = find_chinese_in_text(content, str(py_file))

            if matches:
                files_with_chinese.append({
                    'file_path': str(py_file),
                    'matches': matches[:5]  # 只显示前5个匹配作为示例
                })
                total_chinese_files += 1

            if (i + 1) % 50 == 0:
                print(f"已检查 {i + 1} 个文件...")

        except Exception as e:
            print(f"检查文件 {py_file} 时出错: {e}")

    return files_with_chinese, total_chinese_files

def main():
    root_dir = "/data2/wuyz/ai-engineer-benchmark-dev-react-agent"

    print("=" * 80)
    print("详细检查Python文件中的中文字符")
    print("=" * 80)

    # 先做快速统计
    files_with_chinese, total_chinese = check_python_files_detailed(root_dir, max_files=50)

    print("\n📊 检查结果统计:")
    print(f"  检查的前50个文件中，含有中文的文件: {len(files_with_chinese)} 个")
    print(f"  总计发现中文字符的行数: {sum(len(f['matches']) for f in files_with_chinese)} 行")

    if files_with_chinese:
        print("\n🔍 详细的中文字符发现情况:")
        for file_info in files_with_chinese[:5]:  # 只显示前5个文件作为示例
            print(f"\n📄 文件: {file_info['file_path']}")
            print(f"   发现 {len(file_info['matches'])} 行含有中文字符:")

            for match in file_info['matches']:
                chinese_text = ''.join(match['chinese_chars'])
                print(f"   第{match['line_number']}行: {match['line_content'][:60]}{'...' if len(match['line_content']) > 60 else ''}")
                print(f"      中文字符: '{chinese_text}'")

    print("\n💡 建议:")
    print("  1. 建议将中文注释替换为英文注释")
    print("  2. 字符串中的中文内容可能需要保留（视具体需求而定）")
    print("  3. 可以使用自动化工具进行批量翻译")

if __name__ == "__main__":
    main()
