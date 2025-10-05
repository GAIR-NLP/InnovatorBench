#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查项目中所有Python文件是否含有中文字符
"""

import os
import glob
import re
from pathlib import Path

def contains_chinese(text):
    """
    检查文本是否包含中文字符
    中文字符的Unicode范围主要是 0x4e00-0x9fff
    """
    # 匹配中文字符的正则表达式
    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff]')

    # 排除一些常见的中文标点符号
    text_without_punctuation = re.sub(r'[，。！？；：""''（）【】《》""''\u3000-\u303f]', '', text)

    return bool(chinese_pattern.search(text_without_punctuation))

def check_python_files_for_chinese(root_dir):
    """
    递归检查目录下的所有Python文件是否含有中文
    """
    python_files = []

    # 递归查找所有.py文件
    for py_file in Path(root_dir).rglob('*.py'):
        if py_file.is_file():
            python_files.append(py_file)

    files_with_chinese = []
    files_without_chinese = []
    error_files = []

    print(f"找到 {len(python_files)} 个Python文件，开始检查中文字符...")

    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if contains_chinese(content):
                files_with_chinese.append(str(py_file))
            else:
                files_without_chinese.append(str(py_file))

        except Exception as e:
            error_files.append((str(py_file), str(e)))

    return files_with_chinese, files_without_chinese, error_files

def main():
    root_dir = "/data2/wuyz/ai-engineer-benchmark-dev-react-agent"

    print("=" * 60)
    print("检查Python文件中的中文字符")
    print("=" * 60)

    files_with_chinese, files_without_chinese, error_files = check_python_files_for_chinese(root_dir)

    print(f"\n📊 检查结果统计:")
    print(f"  含有中文的文件: {len(files_with_chinese)} 个")
    print(f"  不含中文的文件: {len(files_without_chinese)} 个")
    print(f"  读取错误的文件: {len(error_files)} 个")

    if files_with_chinese:
        print(f"\n🔍 含有中文的文件列表:")
        for file_path in files_with_chinese:
            print(f"  - {file_path}")

    if error_files:
        print(f"\n❌ 读取错误的文件:")
        for file_path, error in error_files:
            print(f"  - {file_path}: {error}")

    print(f"\n✅ 检查完成！")
    print(f"📝 建议: 建议将含有中文注释的代码替换为英文注释，以保持代码的国际化")

if __name__ == "__main__":
    main()
