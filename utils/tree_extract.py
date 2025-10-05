import json
import os
import re
import sys
from typing import Any, Dict, List, Union, Optional

JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_for_filename(value: Any) -> str:
    """Sanitize any value to a safe filename component."""
    text = str(value) if value is not None else "unknown"
    text = text.strip()
    # Replace path separators
    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")
    # Remove any characters not in safe set
    text = _SAFE_FILENAME_CHARS.sub("_", text)
    # Collapse repeats and trim
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _ensure_unique_path(path: str) -> str:
    """If path exists, append a numeric suffix to make it unique."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    index = 1
    while True:
        candidate = f"{base}_{index}{ext}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


def _save_node(node: Dict[str, Any], output_dir: str) -> Optional[str]:
    """Save a single node dict (excluding 'children') as JSON. Returns saved path or None."""
    if not isinstance(node, dict):
        return None
    if "id" not in node or "timestamp" not in node:
        return None

    record: Dict[str, Any] = {k: v for k, v in node.items() if k != "children"}

    node_id = _sanitize_for_filename(record.get("id"))
    ts = _sanitize_for_filename(record.get("timestamp"))

    filename = f"{node_id}_{ts}.json"
    os.makedirs(output_dir, exist_ok=True)
    path = _ensure_unique_path(os.path.join(output_dir, filename))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return path


essential_keys = {"id", "timestamp", "children"}


def _walk_and_extract(obj: JsonType, output_dir: str, counters: Dict[str, int]) -> None:
    """Recursively walk any JSON-like object and export nodes found."""
    if isinstance(obj, dict):
        saved = _save_node(obj, output_dir)
        if saved:
            counters["saved"] = counters.get("saved", 0) + 1
        # Recurse into all values (including 'children' if present)
        for v in obj.values():
            _walk_and_extract(v, output_dir, counters)
    elif isinstance(obj, list):
        for item in obj:
            _walk_and_extract(item, output_dir, counters)


def main(argv: List[str]) -> int:
    # Determine defaults relative to this script location
    repo_root = os.path.abspath(os.path.dirname(__file__))
    default_input_path = os.path.join(repo_root, "tree_data.json")
    default_output_dir = os.path.join(repo_root, "nodes")

    # Allow optional CLI overrides: python tree_extract.py [input_json] [output_dir]
    input_path = os.path.abspath(argv[1]) if len(argv) >= 2 else default_input_path
    output_dir = os.path.abspath(argv[2]) if len(argv) >= 3 else default_output_dir

    if not os.path.exists(input_path):
        print(f"[ERROR] 找不到输入文件: {input_path}")
        return 1

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 解析失败: {e}")
        return 1

    counters: Dict[str, int] = {"saved": 0}
    _walk_and_extract(data, output_dir, counters)

    print(f"[DONE] 已导出节点: {counters['saved']} 个 -> {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
