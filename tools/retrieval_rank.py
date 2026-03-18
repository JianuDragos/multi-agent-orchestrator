import re
from pathlib import Path

IGNORED_DIRS_DEFAULT = {
    ".git", ".venv", "__pycache__", "node_modules", "runs", "memory", "dist", "build"
}

SOURCE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".htm", ".jinja", ".j2",
    ".css", ".scss", ".sql", ".java", ".go", ".rs", ".php", ".rb", ".cs", ".cpp", ".c"
}

GENERIC_BASENAMES = {
    "app", "config", "database", "models", "index", "base", "main", "utils", "helpers"
}


def _tokenize_keywords(text: str):
    return [
        t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", str(text).lower())
        if len(t) >= 3
    ]


def _tokenize_path(path_text: str):
    return [
        t for t in re.split(r"[/_.\-]+", str(path_text).lower())
        if t and len(t) >= 3
    ]


def collect_relevant_files_scored(root: str, keywords: list, max_files: int = 10, ignored_dirs=None):
    root_path = Path(root).resolve()
    ignored = set(ignored_dirs or IGNORED_DIRS_DEFAULT)

    norm_keywords = []
    for kw in keywords or []:
        kw = str(kw).strip().lower()
        if kw and kw not in norm_keywords:
            norm_keywords.append(kw)

    ranked = []

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored for part in path.parts):
            continue

        rel = str(path.relative_to(root_path)).replace("\\", "/")
        rel_lower = rel.lower()
        stem = path.stem.lower()

        path_tokens = set(_tokenize_path(rel_lower))
        stem_tokens = set(_tokenize_keywords(stem))

        score = 0
        matched = set()

        for kw in norm_keywords:
            kw_parts = set(_tokenize_keywords(kw))

            if kw == stem or kw in stem_tokens:
                score += 12
                matched.add(kw)
                continue

            if kw in path_tokens:
                score += 7
                matched.add(kw)
                continue

            if kw in rel_lower:
                score += 3
                matched.add(kw)

            for part in kw_parts:
                if part in stem_tokens:
                    score += 5
                    matched.add(part)
                elif part in path_tokens:
                    score += 2
                    matched.add(part)

        score += min(10, len(matched) * 2)

        if path.suffix.lower() in SOURCE_EXTS:
            score += 1

        if stem in GENERIC_BASENAMES and score > 0:
            score -= 1

        if score > 0:
            ranked.append({
                "score": score,
                "path": path,
                "matched_keywords": sorted(matched),
                "matched_count": len(matched),
            })

    ranked.sort(key=lambda item: (-item["score"], str(item["path"]).lower()))
    return ranked[:max_files]
