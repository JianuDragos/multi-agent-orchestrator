from pathlib import Path
import ast
import re


IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "runs",
    "memory",
    "dist",
    "build",
}


def build_project_tree(root: str, max_depth: int = 5) -> str:
    root_path = Path(root).resolve()
    lines = [root_path.name + "/"]

    def walk(path: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return

        try:
            entries = sorted(
                [p for p in path.iterdir() if p.name not in IGNORED_DIRS],
                key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            return

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")

            if entry.is_dir():
                next_prefix = prefix + ("    " if is_last else "│   ")
                walk(entry, next_prefix, depth + 1)

    walk(root_path)
    return "\n".join(lines)


def _py_skeleton(content: str) -> str:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ""

    out = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(ast.get_source_segment(content, node) or "")
        elif isinstance(node, ast.ClassDef):
            out.append(f"class {node.name}:")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in child.args.args]
                    prefix = "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
                    out.append(f"    {prefix} {child.name}({', '.join(args)}): ...")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            out.append(f"{prefix} {node.name}({', '.join(args)}): ...")

    return "\n".join(line for line in out if line.strip())


def _js_ts_skeleton(content: str) -> str:
    lines = content.splitlines()
    kept = []

    patterns = (
        r"^\s*import\s+",
        r"^\s*export\s+",
        r"^\s*function\s+\w+",
        r"^\s*async\s+function\s+\w+",
        r"^\s*const\s+\w+\s*=\s*\(",
        r"^\s*let\s+\w+\s*=\s*\(",
        r"^\s*var\s+\w+\s*=\s*\(",
        r"^\s*class\s+\w+",
        r"^\s*router\.",
        r"^\s*app\.",
        r"^\s*document\.",
        r"^\s*window\.",
    )

    for line in lines:
        if any(re.search(pat, line) for pat in patterns):
            kept.append(line)

    return "\n".join(kept)


def _html_skeleton(content: str) -> str:
    lines = content.splitlines()
    kept = []

    patterns = (
        r"<form\b",
        r"<button\b",
        r"<input\b",
        r"<select\b",
        r"<a\b",
        r"id=",
        r"class=",
        r"url_for\(",
        r"{% extends",
        r"{% block",
        r"{% for",
        r"{% if",
    )

    for line in lines:
        if any(re.search(pat, line) for pat in patterns):
            kept.append(line)

    return "\n".join(kept)


def read_file_safe(path: Path, max_chars: int = 12000) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"[Could not read file: {e}]"

    if len(content) <= max_chars:
        return content

    suffix = path.suffix.lower()

    if suffix == ".py":
        skeleton = _py_skeleton(content)
        if skeleton.strip():
            return skeleton + f"\n\n...[python skeleton extracted from {len(content)} chars]..."

    if suffix in {".js", ".ts", ".jsx", ".tsx"}:
        skeleton = _js_ts_skeleton(content)
        if skeleton.strip():
            return skeleton + f"\n\n...[js/ts skeleton extracted from {len(content)} chars]..."

    if suffix in {".html", ".htm", ".jinja", ".j2"}:
        skeleton = _html_skeleton(content)
        if skeleton.strip():
            return skeleton + f"\n\n...[html skeleton extracted from {len(content)} chars]..."

    head = content[: max_chars // 2]
    tail = content[-max_chars // 2 :]
    return head + "\n\n...[TRUNCATED]...\n\n" + tail


def _path_tokens(root_path: Path, path: Path):
    rel = str(path.relative_to(root_path)).lower()
    parts = re.split(r"[/_.\\-]+", rel)
    return {p for p in parts if p}


def collect_relevant_files(root: str, keywords=None, max_files: int = 8):
    if keywords is None:
        keywords = []

    root_path = Path(root).resolve()
    candidates = []
    seen = set()

    normalized_keywords = []
    for kw in keywords:
        kw = str(kw).strip().lower()
        if len(kw) >= 2:
            normalized_keywords.append(kw)

    for path in root_path.rglob("*"):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue

        rel = str(path.relative_to(root_path)).replace("\\", "/")
        rel_lower = rel.lower()
        name_lower = path.name.lower()
        tokens = _path_tokens(root_path, path)

        score = 0

        for kw in normalized_keywords:
            if kw == name_lower:
                score += 10
            if kw in tokens:
                score += 7
            elif kw in rel_lower:
                score += 2

        if any(t in tokens for t in {"route", "routes", "controller", "service"}):
            score += 1
        if any(t in tokens for t in {"main", "app", "index"}):
            score += 1
        if any(t in tokens for t in {"html", "js", "py"}):
            score += 1

        if score > 0 and rel not in seen:
            candidates.append((score, path))
            seen.add(rel)

    candidates.sort(key=lambda item: (-item[0], str(item[1])))

    selected = []
    for _, path in candidates:
        selected.append(path)
        if len(selected) >= max_files:
            break

    return selected
