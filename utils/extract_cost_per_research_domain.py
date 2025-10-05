import os
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

@dataclass
class MetricTotals:
    sum_input_tokens: int = 0
    sum_output_tokens: int = 0
    sum_total_tokens: int = 0
    count_input_tokens: int = 0
    count_output_tokens: int = 0
    count_total_tokens: int = 0

    def add(self, input_tokens: Optional[int], output_tokens: Optional[int], total_tokens: Optional[int]) -> None:
        if input_tokens is not None:
            self.sum_input_tokens += input_tokens
            self.count_input_tokens += 1
        if output_tokens is not None:
            self.sum_output_tokens += output_tokens
            self.count_output_tokens += 1
        if total_tokens is not None:
            self.sum_total_tokens += total_tokens
            self.count_total_tokens += 1

    def merge(self, other: "MetricTotals") -> None:
        self.sum_input_tokens += other.sum_input_tokens
        self.sum_output_tokens += other.sum_output_tokens
        self.sum_total_tokens += other.sum_total_tokens
        self.count_input_tokens += other.count_input_tokens
        self.count_output_tokens += other.count_output_tokens
        self.count_total_tokens += other.count_total_tokens

    def averages(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        avg_in = (self.sum_input_tokens / self.count_input_tokens) if self.count_input_tokens else None
        avg_out = (self.sum_output_tokens / self.count_output_tokens) if self.count_output_tokens else None
        avg_total = (self.sum_total_tokens / self.count_total_tokens) if self.count_total_tokens else None
        return avg_in, avg_out, avg_total


def safe_to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int,)):
            return int(value)
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
            # Support strings like "123" or "123.0"
            if value.isdigit() or (value.replace(".", "", 1).isdigit() and value.count(".") <= 1):
                return int(float(value))
        return None
    except Exception:
        return None


def extract_usage_from_file(file_path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] 无法解析 JSON: {file_path} -> {e}", file=sys.stderr)
        return None, None, None

    usage = None
    try:
        usage = data.get("response", {}).get("usage")
    except Exception:
        usage = None

    if not isinstance(usage, dict):
        return None, None, None

    input_tokens = safe_to_int(usage.get("input_tokens"))
    output_tokens = safe_to_int(usage.get("output_tokens"))
    total_tokens = safe_to_int(usage.get("total_tokens"))
    return input_tokens, output_tokens, total_tokens


def process_directory(dir_path: Path) -> Tuple[MetricTotals, int, Dict[str, MetricTotals]]:
    totals = MetricTotals()
    domain_totals: Dict[str, MetricTotals] = {}
    json_files = sorted([p for p in dir_path.glob("*.json") if p.is_file()])
    for fp in json_files:
        in_tok, out_tok, tot_tok = extract_usage_from_file(fp)
        totals.add(in_tok, out_tok, tot_tok)

        # 推断研究领域（research domain）：文件名前缀到第一个下划线
        name = fp.name
        domain = name.split("_", 1)[0] if "_" in name else "unknown"
        bucket = domain_totals.setdefault(domain, MetricTotals())
        bucket.add(in_tok, out_tok, tot_tok)

    return totals, len(json_files), domain_totals


def format_number(n: Optional[float]) -> str:
    if n is None:
        return "-"
    if isinstance(n, float):
        return f"{n:.2f}"
    return f"{n}"


def print_section_header(title: str) -> None:
    print("".ljust(2) + "=" * 60)
    print("".ljust(2) + title)
    print("".ljust(2) + "=" * 60)


def print_totals(label: str, totals: MetricTotals, file_count: Optional[int]) -> None:
    avg_in, avg_out, avg_total = totals.averages()
    print(f"  [{label}]")
    if file_count is not None:
        print(f"    文件数: {file_count}")
    print(
        "    求和  -> input_tokens: {inp}, output_tokens: {out}, total_tokens: {tot}".format(
            inp=totals.sum_input_tokens,
            out=totals.sum_output_tokens,
            tot=totals.sum_total_tokens,
        )
    )
    print(
        "    均值  -> input_tokens: {inp}, output_tokens: {out}, total_tokens: {tot}".format(
            inp=format_number(avg_in),
            out=format_number(avg_out),
            tot=format_number(avg_total),
        )
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    benchmark_file = "benchmark.txt"
    if not os.path.exists(benchmark_file):
        print(f"错误: 文件 '{benchmark_file}' 不存在")
        return
    
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        workspace_dirs = [line.strip() for line in f if line.strip()]
    
    if not workspace_dirs:
        print("benchmark.txt 中没有有效的 workspace_dir")
        return

    overall = MetricTotals()
    overall_domain_totals: Dict[str, MetricTotals] = {}

    # 准备 tokens 输出目录
    tokens_base = Path("tokens")
    tokens_base.mkdir(parents=True, exist_ok=True)

    print_section_header("各目录统计")
    # 收集每个任务（目录）的 input/output tokens 总和
    input_per_task = []
    output_per_task = []
    for directory in workspace_dirs:
        
        abs_dir = "/inspire/hdd/project/qproject-fundationmodel/liupengfei-24025/ai_engineer/workspace_backup"
        nodes_dir = os.path.join(abs_dir, directory, "nodes")
        
        dir_totals, file_count, domain_totals = process_directory(Path(nodes_dir))
        print_totals(str(nodes_dir), dir_totals, file_count)

        # 记录每个任务的 tokens 汇总
        input_per_task.append(dir_totals.sum_input_tokens)
        output_per_task.append(dir_totals.sum_output_tokens)

        # 汇总到 overall
        overall.merge(dir_totals)
        for domain, t in domain_totals.items():
            overall_bucket = overall_domain_totals.setdefault(domain, MetricTotals())
            overall_bucket.merge(t)

    print()
    print_section_header("总体统计（合并所有目录）")
    print_totals("ALL", overall, None)

    if overall_domain_totals:
        print()
        print_section_header("按研究领域统计（基于文件名前缀）")
        for domain in sorted(overall_domain_totals.keys()):
            print_totals(domain, overall_domain_totals[domain], None)

    # 写入 tokens/input_token.txt 与 tokens/output_token.txt
    # 规则：前 20 行为前 20 个任务的 tokens 总和，第 21 行为上述 20 个值的平均值（四舍五入为整数）
    top_n = 20
    selected_input = input_per_task[:top_n]
    selected_output = output_per_task[:top_n]

    avg_input = int(round(sum(selected_input) / len(selected_input))) if selected_input else 0
    avg_output = int(round(sum(selected_output) / len(selected_output))) if selected_output else 0

    input_txt = tokens_base / "input_token.txt"
    output_txt = tokens_base / "output_token.txt"
    try:
        with input_txt.open("w", encoding="utf-8") as f_in:
            for v in selected_input:
                f_in.write(f"{v}\n")
            f_in.write(f"{avg_input}\n")
    except Exception as e:
        print(f"[WARN] 写入文件失败: {input_txt} -> {e}", file=sys.stderr)
    try:
        with output_txt.open("w", encoding="utf-8") as f_out:
            for v in selected_output:
                f_out.write(f"{v}\n")
            f_out.write(f"{avg_output}\n")
    except Exception as e:
        print(f"[WARN] 写入文件失败: {output_txt} -> {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


