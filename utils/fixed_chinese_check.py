#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版本：正确检查项目中Python文件中的中文字符
"""

import os
import re
from pathlib import Path

def contains_chinese(text):
    """
    检查文本是否包含中文字符（仅匹配真正的中文汉字，不包括标点符号）
    中文字符的Unicode范围主要是 0x4e00-0x9fff
    """
    # 只匹配中文汉字，不包括标点符号
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

    return bool(chinese_pattern.search(text))

def find_chinese_in_text(text, file_path):
    """
    在文本中查找中文字符，并返回详细的匹配信息
    """
    chinese_pattern = re.compile(r'([\u4e00-\u9fff])')

    matches = []
    lines = text.split('\n')

    for i, line in enumerate(lines, 1):
        chinese_chars = chinese_pattern.findall(line)
        if chinese_chars:
            # 过滤掉纯标点符号的行（但保留包含中文字符的行）
            line_without_chinese = chinese_pattern.sub('', line)
            punctuation_only = re.match(r'^[\s\u3000-\u303f]*$', line_without_chinese)
            if not punctuation_only:
                matches.append({
                    'line_number': i,
                    'line_content': line.strip(),
                    'chinese_chars': chinese_chars,
                    'file_path': file_path
                })

    return matches

def check_python_files_for_chinese(root_dir, sample_size=20):
    """
    检查目录下的所有Python文件是否含有中文字符
    """
    python_files = []

    # 递归查找所有.py文件
    for py_file in Path(root_dir).rglob('*.py'):
        if py_file.is_file() and not any(part.startswith('.') for part in py_file.parts):
            python_files.append(py_file)

    files_with_chinese = []
    total_chinese_files = 0

    print(f"找到 {len(python_files)} 个Python文件，开始检查中文字符...")

    # 检查所有文件
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if contains_chinese(content):
                matches = find_chinese_in_text(content, str(py_file))
                if matches:  # 确保有真正的中文字符
                    files_with_chinese.append({
                        'file_path': str(py_file),
                        'matches': matches
                    })
                    total_chinese_files += 1

        except Exception as e:
            print(f"检查文件 {py_file} 时出错: {e}")

    return files_with_chinese, total_chinese_files

def main():
    root_dir = "/data2/wuyz/ai-engineer-benchmark-dev-react-agent"

    print("=" * 70)
    print("修复版本：检查Python文件中的中文字符")
    print("=" * 70)

    files_with_chinese, total_chinese = check_python_files_for_chinese(root_dir)

    print("\n📊 检查结果统计:")
    print(f"  含有中文字符的文件: {len(files_with_chinese)} 个")
    print(f"  总计发现中文字符的行数: {sum(len(f['matches']) for f in files_with_chinese)} 行")

    if files_with_chinese:
        print("\n🔍 详细的中文字符发现情况:")
        for file_info in files_with_chinese:
            if "/data2/wuyz/ai-engineer-benchmark-dev-react-agent/utils/" in file_info['file_path']:
                continue
            print(f"\n📄 文件: {file_info['file_path']}")
            print(f"   发现 {len(file_info['matches'])} 行含有中文字符:")

            # for match in file_info['matches']:
            #     chinese_text = ''.join(match['chinese_chars'])
            #     print(f"   第{match['line_number']}行: {match['line_content'][:80]}{'...' if len(match['line_content']) > 80 else ''}")
            #     print(f"      中文字符: '{chinese_text}'")

    print("\n💡 建议:")
    print("  1. 建议将中文注释替换为英文注释")
    print("  2. 字符串中的中文内容可能需要保留（视具体需求而定）")
    print("  3. 可以使用自动化工具进行批量翻译")

    # 列出没有中文字符的文件数量
    all_python_files = len(list(Path(root_dir).rglob('*.py')))
    files_without_chinese = all_python_files - len(files_with_chinese)
    print(f"\n📈 完整统计:")
    print(f"  总Python文件数: {all_python_files}")
    print(f"  含有中文字符的文件: {len(files_with_chinese)} ({len(files_with_chinese)/all_python_files*100:.1f}%)")
    print(f"  不含中文字符的文件: {files_without_chinese} ({files_without_chinese/all_python_files*100:.1f}%)")

if __name__ == "__main__":
    main()
