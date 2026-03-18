import json
import re
from pathlib import Path

from tools.ollama_client import OllamaClient
from tools.project_scan import build_project_tree, collect_relevant_files, read_file_safe
from tools.retrieval_rank import collect_relevant_files_scored
from tools.json_utils import extract_json_candidates
from tools.plan_validation import (
    normalize_list,
    normalize_plan_paths,
    validate_coordinator_plan,
    validate_supervisor_plan,
)


IGNORED_DIRS = {".git", ".venv", "__pycache__", "node_modules", "runs", "memory"}


def task_keywords(task: str):
    raw = keyword_guess(task)
    seen = []
    for item in raw:
        if item not in seen:
            seen.append(item)
    return seen[:12]


RETRIEVAL_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "onto",
    "when", "then", "than", "your", "their", "there", "have", "has", "had",
    "will", "would", "should", "could", "without", "within", "about", "after",
    "before", "where", "which", "while", "must", "need", "needs", "only",
    "just", "make", "does", "page", "file", "files", "task", "project",
    "fix", "update", "change"
}


def keyword_guess(task: str):
    tokens = []
    raw = re.findall(r"[a-zA-Z_][a-zA-Z0-9_\-]{2,}", task.lower())

    for tok in raw:
        tok = tok.strip("._-")
        if len(tok) >= 3 and tok not in RETRIEVAL_STOPWORDS:
            tokens.append(tok)

    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append(f"{tokens[i]}_{tokens[i+1]}")

    return list(dict.fromkeys(tokens + bigrams))[:18]


def get_keywords_agentic(client, task: str, tree: str):
    fallback = keyword_guess(task)

    prompt = f"""
You are a retrieval helper.
Generate search keywords for finding relevant files in a codebase.

Return exactly one JSON object with this schema:
{{
  "keywords": ["kw1", "kw2", "kw3"]
}}

Rules:
- Output JSON only.
- Do not output markdown.
- Do not explain.
- Prefer short file-search terms.
- Include domain terms from the task.
- Include likely code terms from the tree when relevant.
- Return 8 to 15 keywords.
- Do not invent technologies not suggested by the task or tree.

TASK:
{task}

PROJECT TREE:
{tree}
""".strip()

    try:
        raw = client.generate("retriever", prompt)
        candidates = extract_json_candidates(raw)

        for entry in reversed(candidates):
            obj = entry.get("object", {})
            kws = obj.get("keywords", [])
            kws = normalize_list(kws)
            kws = [
                str(x).strip().lower()
                for x in kws
                if str(x).strip() and len(str(x).strip()) >= 3
            ]
            if kws:
                return list(dict.fromkeys(kws))[:18]
    except Exception as exc:
        print(f"[retrieval keyword fallback] {type(exc).__name__}: {exc}")

    return fallback


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "onto",
    "when", "then", "than", "your", "their", "there", "have", "has", "had",
    "will", "would", "should", "could", "without", "within", "about", "after",
    "before", "where", "which", "while", "must", "need", "needs", "only",
    "just", "make", "does", "page", "file", "files", "task"
}

NOVELTY_PROSE_STOPWORDS = {
    "actual", "actually", "blocking", "blocked", "cause", "causes", "causing",
    "check", "checks", "checking", "code", "logic", "issue", "issues", "root",
    "current", "currently", "verify", "verifies", "verifying", "ensure",
    "ensures", "ensuring", "behavior", "correct", "correctly", "request",
    "requests", "response", "responses", "submit", "submits", "submission",
    "form", "forms", "route", "routes", "handler", "handlers", "endpoint",
    "endpoints", "problem", "problems", "value", "values", "action", "actions",
    "button", "buttons", "template", "templates", "functionality",
    "contains", "contain", "including", "include", "inside", "locate",
    "review", "modify", "change", "test", "maintain", "maintains", "remain",
    "remains", "functional", "consistent", "targeting", "points", "point"
}


def _tokenize_text(text: str):
    return [
        t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", str(text).lower())
        if t not in STOPWORDS and len(t) >= 3
    ]


def _tokenize_path(path: str):
    return [
        t for t in re.split(r"[/_.\-]+", str(path).lower())
        if t and t not in STOPWORDS and len(t) >= 3
    ]


def _flatten_plan_text(value):
    out = []

    def walk(v):
        if isinstance(v, dict):
            for item in v.values():
                walk(item)
        elif isinstance(v, list):
            for item in v:
                walk(item)
        elif isinstance(v, str):
            out.append(v)

    walk(value)
    return "\n".join(out)


def _task_alignment_problems(plan, task):
    if not isinstance(plan, dict):
        return []

    task_tokens = set(_tokenize_text(task))
    if not task_tokens:
        return []

    plan_text = _flatten_plan_text(plan)
    plan_tokens = set(_tokenize_text(plan_text))
    overlap = task_tokens.intersection(plan_tokens)

    if not overlap:
        return ["plan appears off-task: no meaningful task keywords appear in the plan"]

    return []


def context_relevance_problems(plan: dict, used_context_files: list[str], used_context_score_map: dict, used_context_match_count_map: dict):
    if not isinstance(plan, dict):
        return []

    if not used_context_files or not used_context_score_map:
        return []

    files_to_modify = {str(x).strip() for x in normalize_list(plan.get("files_to_modify", []))}
    files_to_avoid = {str(x).strip() for x in normalize_list(plan.get("files_to_avoid", []))}
    evidence_files = set()

    evidence = plan.get("evidence", [])
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                f = str(item.get("file", "")).strip()
                if f:
                    evidence_files.add(f)

    addressed = files_to_modify | files_to_avoid | evidence_files
    top_score = max(used_context_score_map.get(path, 0) for path in used_context_files)
    if top_score < 8:
        return []

    generic_basenames = {"app", "config", "database", "models", "index", "base", "main", "utils", "helpers"}
    problems = []

    for path in used_context_files:
        score = used_context_score_map.get(path, 0)
        matched_count = used_context_match_count_map.get(path, 0)
        stem = Path(path).stem.lower()

        if stem in generic_basenames:
            continue

        # General rule:
        # only flag ignored files that were retrieved strongly AND matched multiple distinct keywords
        if score >= max(8, top_score - 2) and matched_count >= 2 and path not in addressed:
            problems.append(f"high-priority multi-signal retrieved file not addressed: {path}")

    return problems[:2]


TECH_NOVELTY_TERMS = {
    "ajax", "fetch", "axios", "websocket", "graphql", "stripe", "paypal",
    "redis", "celery", "kafka", "rabbitmq", "docker", "nginx", "s3",
    "jwt", "oauth", "grpc", "redux", "zustand", "pinia", "socketio",
    "prisma", "typeorm", "sequelize", "alembic", "flyway", "supabase",
    "firebase", "nextauth", "passport", "bullmq", "airflow"
}


def unsupported_novelty_problems(plan: dict, task: str, used_context_files: list[str], relevant_context: str):
    if not isinstance(plan, dict):
        return []

    plan_text = _flatten_plan_text(plan).lower()
    allowed_text_parts = [str(task).lower(), " ".join(used_context_files).lower(), str(relevant_context).lower()]

    for f in normalize_list(plan.get("files_to_modify", [])):
        allowed_text_parts.append(str(f).lower())

    for f in normalize_list(plan.get("files_to_avoid", [])):
        allowed_text_parts.append(str(f).lower())

    evidence = plan.get("evidence", [])
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                f = str(item.get("file", "")).strip()
                if f:
                    allowed_text_parts.append(f.lower())

    allowed_text = "\n".join(allowed_text_parts)

    suspicious = [term for term in sorted(TECH_NOVELTY_TERMS) if term in plan_text and term not in allowed_text]

    if suspicious:
        return [f"plan introduces unsupported technical mechanisms not grounded in task/context: {', '.join(suspicious[:6])}"]

    return []


def list_all_project_files(root: Path):
    files = []
    for path in root.rglob("*"):
        if path.is_file():
            if any(part in IGNORED_DIRS for part in path.parts):
                continue
            files.append(str(path.relative_to(root)).replace("\\", "/"))
    return sorted(files)


def validate_worker_plan(plan, allowed_files, task=None):
    problems = []

    if not isinstance(plan, dict):
        return ["Worker plan is not a JSON object"]

    plan = normalize_plan_paths(plan)

    implementation_summary = str(plan.get("implementation_summary", "")).strip()
    if not implementation_summary:
        problems.append("implementation_summary is missing or empty")

    aider_task = str(plan.get("aider_task", "")).strip()
    if not aider_task:
        problems.append("aider_task is missing or empty")

    files_to_modify = normalize_list(plan.get("files_to_modify", []))
    if not files_to_modify:
        problems.append("files_to_modify is missing or empty")

    evidence = plan.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        problems.append("evidence is missing or empty")

    allowed = set(allowed_files)
    for f in files_to_modify:
        if f not in allowed:
            problems.append(f"Worker files_to_modify contains non-approved path: {f}")

    evidence_files = []
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                problems.append("evidence contains a non-object item")
                continue
            f = str(item.get("file", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not f:
                problems.append("evidence item is missing file")
                continue
            if not reason:
                problems.append(f"evidence item is missing reason for file: {f}")
            if f not in allowed:
                problems.append(f"Worker evidence contains non-approved path: {f}")
            evidence_files.append(f)

    for f in files_to_modify:
        if f not in evidence_files:
            problems.append(f"files_to_modify missing matching evidence: {f}")

    problems.extend(_task_alignment_problems(plan, task))
    return problems


def select_best_candidate(raw_text, expected_keys, validator, validation_context, task, label):
    candidates = extract_json_candidates(raw_text)
    ranked = []

    for entry in candidates:
        obj = normalize_plan_paths(entry["object"])
        present_key_count = sum(1 for key in expected_keys if key in obj)
        validation_problems = validator(obj, validation_context, task)

        score = (
            len(validation_problems),
            -present_key_count,
            -len(obj),
            -entry["index"],
        )

        ranked.append({
            "score": score,
            "object": obj,
            "problems": validation_problems,
            "index": entry["index"],
        })

    ranked.sort(key=lambda item: item["score"])
    best = ranked[0]

    print(f"\n=== {label} candidate selection ===")
    print(f"Found {len(ranked)} JSON candidate(s)")
    print(f"Selected candidate index: {best['index']}")
    print(f"Selected candidate validation problem count: {len(best['problems'])}")

    return best["object"]


def build_relevant_context(relevant_files, root: Path, max_file_chars: int, max_total_context_chars: int):
    blocks = []
    used_files = []
    total_chars = 0

    for path in relevant_files:
        rel = path.relative_to(root)
        content = read_file_safe(path, max_chars=max_file_chars)
        block = f"=== FILE: {rel} ===\n{content}\n"

        if blocks and total_chars + len(block) > max_total_context_chars:
            break

        if not blocks and len(block) > max_total_context_chars:
            block = block[:max_total_context_chars]

        blocks.append(block)
        used_files.append(str(rel).replace("\\", "/"))
        total_chars += len(block)

    return "\n".join(blocks), used_files, total_chars


def format_raw_response_for_display(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return text

    try:
        candidates = extract_json_candidates(text)
        if candidates:
            best = max(
                candidates,
                key=lambda item: (len(item["object"]), item["index"])
            )
            return json.dumps(best["object"], indent=2, ensure_ascii=False)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return f"[display parse warning: {type(exc).__name__}]\n{text[:2000]}"

    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]

    return text


def context_grounding_problems(plan: dict, used_context_files: list[str]):
    problems = []

    if not isinstance(plan, dict):
        return ["plan is not a JSON object for grounding validation"]

    used = {str(x).strip() for x in used_context_files if str(x).strip()}

    files_to_modify = [str(x).strip() for x in plan.get("files_to_modify", []) if str(x).strip()]
    evidence_files = []

    evidence = plan.get("evidence", [])
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                f = str(item.get("file", "")).strip()
                if f:
                    evidence_files.append(f)

    missing_modify = [f for f in files_to_modify if f not in used]
    missing_evidence = [f for f in evidence_files if f not in used]

    for f in missing_modify:
        problems.append(f"files_to_modify not grounded in used context: {f}")

    for f in missing_evidence:
        problems.append(f"evidence file not grounded in used context: {f}")

    return problems


def print_grounding_debug(label: str, plan: dict, used_context_files: list[str]):
    evidence = plan.get("evidence", [])
    evidence_files = []

    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                f = str(item.get("file", "")).strip()
                if f:
                    evidence_files.append(f)

    files_to_modify = [str(x) for x in plan.get("files_to_modify", [])]
    files_to_avoid = [str(x) for x in plan.get("files_to_avoid", [])]

    print(f"\n=== {label} grounding ===")
    print("Used context files:")
    for f in used_context_files:
        print("-", f)

    print("Evidence files:")
    if evidence_files:
        for f in evidence_files:
            print("-", f)
    else:
        print("(none)")

    print("Files to modify:")
    if files_to_modify:
        for f in files_to_modify:
            print("-", f)
    else:
        print("(none)")

    print("Files to avoid:")
    if files_to_avoid:
        for f in files_to_avoid:
            print("-", f)
    else:
        print("(none)")

    missing_from_context = [f for f in files_to_modify if f not in used_context_files]
    if missing_from_context:
        print("Modify files not present in used context:")
        for f in missing_from_context:
            print("-", f)

    evidence_not_in_context = [f for f in evidence_files if f not in used_context_files]
    if evidence_not_in_context:
        print("Evidence files not present in used context:")
        for f in evidence_not_in_context:
            print("-", f)


def run_json_stage(client, role, prompt, expected_keys, validator, validation_context, task, label):
    raw = client.generate(role, prompt)

    print(f"\n=== {label} raw response (filtered) ===")
    print(format_raw_response_for_display(raw))

    plan = select_best_candidate(
        raw_text=raw,
        expected_keys=expected_keys,
        validator=validator,
        validation_context=validation_context,
        task=task,
        label=label,
    )

    print(f"\n=== {label} parsed JSON ===")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    problems = validator(plan, validation_context, task)

    print(f"\n=== {label} validation ===")
    if problems:
        for p in problems:
            print("-", p)
    else:
        print(f"No {label.lower()} validation problems detected.")

    return raw, plan, problems


def build_supervisor_prompt(task, task_kw, real_files, coordinator_plan, coordinator_validation_problems, used_context_files_text):
    return f"""
You are a specialized JSON generator acting as a skeptical supervisor in a multi-agent coding system.
You are NOT a chat assistant.
Do not use chain-of-thought.
Do not write "Thinking..." or explanations.
Start your response with {{ and end with }}.

Your job is to find reasons the coordinator plan may fail, correct it if needed, and return exactly one JSON object.
Prefer rejecting weakly grounded or incomplete plans rather than agreeing too easily.

PRIMARY TASK TO SOLVE:
{task}

TASK KEYWORDS:
{json.dumps(task_kw, ensure_ascii=False)}

You must review the plan against the PRIMARY TASK above exactly as written.

All real project files:
{json.dumps(real_files, ensure_ascii=False)}

Coordinator plan:
{json.dumps(coordinator_plan, ensure_ascii=False)}

Coordinator validation problems:
{json.dumps(coordinator_validation_problems, ensure_ascii=False)}

Context files the coordinator had available:
{used_context_files_text}

CRITICAL REMINDER BEFORE YOU ANSWER:
PRIMARY TASK: {task}
KEYWORDS: {json.dumps(task_kw, ensure_ascii=False)}
Output ONLY JSON. Do not write "Thinking..." or explanations.

Return exactly one valid JSON object with this exact schema:
{{
  "approved": true,
  "worker": "code or ui",
  "reason": "short text",
  "files_to_modify": ["real/project/file1", "real/project/file2"],
  "files_to_avoid": ["real/project/file3"],
  "evidence": [{{"file": "real/project/file1", "reason": "why this file is relevant"}}],
  "constraints": ["constraint1", "constraint2"],
  "corrected_steps": ["step1", "step2", "step3"]
}}

Hard rules:
- Output JSON only.
- Do not output markdown.
- Do not use code fences.
- Do not output explanations.
- Do not output analysis.
- Do not output thinking.
- Do not write any text before the opening {{.
- Do not write any text after the closing }}.
- The response must start with {{ and end with }}.
- Use ONLY file paths from "All real project files".
- Prefer minimal safe changes.
- Reject invented paths.
- Reject unnecessarily broad changes.
- If the task mentions visible UI elements or page behavior (button, form, input, navbar, modal, page, layout, template), expect the final plan to consider both rendering files and behavior files when appropriate.
- If the task mentions paired or opposite actions (increase/decrease, open/close, show/hide, enable/disable), reject plans that clearly cover only one side of the behavior.
- Reject plans that place likely rendering files (.html, .htm, .jinja, .j2, template files) in files_to_avoid without clear evidence.
- Do not add unrelated enhancements or adjacent features.
- Do not include orchestration metadata, path normalization instructions, or repository-management steps.
- STRICT LIMIT: Maximum 3 files_to_modify unless absolutely necessary.
- If more than 3 files are truly required, explain why in "reason".
- Every schema field is required.
- Every file in files_to_modify must be justified in evidence.
- Every evidence item must use a real file path and a short reason.
- Always include "constraints"; use [] if there are no constraints.
- "approved" means whether your FINAL supervisor plan is safe to execute.
- If you corrected the coordinator plan and your corrected plan is safe, set "approved" to true.
- Set "approved" to false only if execution should stop and a new planning pass is required.
""".strip()


def main():
    client = OllamaClient()
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    limits = config.get("limits", {})
    paths_cfg = config.get("paths", {})

    max_file_chars = int(limits.get("max_file_chars", 12000))
    max_context_files = int(limits.get("max_context_files", 4))
    max_total_context_chars = int(limits.get("max_total_context_chars", 40000))
    strict_worker_context_grounding = bool(config.get("strict_worker_context_grounding", True))

    runs_dir = Path(paths_cfg.get("runs_dir", "runs"))
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("=== Multi-Agent Orchestrator ===")
    project_path = input("Enter project path: ").strip()
    task = input("Enter task: ").strip()

    if not project_path:
        print("No project path provided.")
        return

    if not task:
        print("No task provided.")
        return

    root = Path(project_path).resolve()
    if not root.exists():
        print(f"Project path does not exist: {root}")
        return

    print("\n[Context] Building project tree...")
    tree = build_project_tree(str(root))
    real_files = list_all_project_files(root)

    keywords = get_keywords_agentic(client, task, tree)
    task_kw = task_keywords(task)

    print("\n=== Retrieval keywords ===")
    for kw in keywords:
        print("-", kw)

    ranked_relevant_files = collect_relevant_files_scored(
        str(root),
        keywords=keywords,
        max_files=max_context_files,
    )
    relevant_files = [item["path"] for item in ranked_relevant_files]
    relevant_score_map = {
        str(item["path"].relative_to(root)).replace("\\", "/"): item["score"]
        for item in ranked_relevant_files
    }
    relevant_match_count_map = {
        str(item["path"].relative_to(root)).replace("\\", "/"): item["matched_count"]
        for item in ranked_relevant_files
    }

    print("\n=== Relevant files selected ===")
    for item in ranked_relevant_files:
        rel = item["path"].relative_to(root)
        print(f"- {rel}  [score={item['score']}, matched={item['matched_count']}]")

    relevant_context, used_context_files, total_context_chars = build_relevant_context(
        relevant_files=relevant_files,
        root=root,
        max_file_chars=max_file_chars,
        max_total_context_chars=max_total_context_chars,
    )
    used_context_score_map = {
        path: relevant_score_map.get(path, 0)
        for path in used_context_files
    }
    used_context_match_count_map = {
        path: relevant_match_count_map.get(path, 0)
        for path in used_context_files
    }

    print("\n=== Context budget ===")
    print(f"Used files in context: {len(used_context_files)}")
    print(f"Total context chars: {total_context_chars}")
    print(f"Max total context chars: {max_total_context_chars}")

    real_files_text = "\n".join(real_files)
    used_context_files_text = json.dumps(used_context_files, ensure_ascii=False)

    coordinator_prompt_base = f"""
You are a specialized JSON generator acting as the coordinator in a multi-agent coding system.
You are NOT a chat assistant.
Do not use chain-of-thought.
Do not write "Thinking..." or explanations.
Start your response with {{ and end with }}.

Your job is to produce a narrow, grounded implementation plan for exactly the user's task.

Project root:
{root}

PRIMARY TASK TO SOLVE:
{task}

TASK KEYWORDS:
{json.dumps(task_kw, ensure_ascii=False)}

You must solve the PRIMARY TASK above exactly as written.

Project tree:
{tree}

All real project files:
{real_files_text}

Context files you were actually given:
{used_context_files_text}

Relevant file contents:
{relevant_context}

CRITICAL REMINDER BEFORE YOU ANSWER:
PRIMARY TASK: {task}
KEYWORDS: {json.dumps(task_kw, ensure_ascii=False)}
Output ONLY JSON. Do not write "Thinking..." or explanations.

Return ONLY one valid JSON object with this exact schema:
{{
  "task_type": "code or ui or mixed",
  "probable_cause": "short text",
  "files_to_modify": ["real/project/file1", "real/project/file2"],
  "files_to_avoid": ["real/project/file3"],
  "evidence": [{{"file": "real/project/file1", "reason": "why this file is relevant"}}],
  "steps": ["step1", "step2", "step3"],
  "risks": ["risk1", "risk2"]
}}

Rules:
- Use ONLY real file paths from "All real project files".
- Prefer files from "Context files you were actually given".
- Do not invent filenames.
- Stay strictly on the requested task.
- If the task mentions visible UI elements or page behavior (button, form, input, navbar, modal, page, layout, template), consider both rendering files and behavior files.
- If the task mentions paired or opposite actions (increase/decrease, open/close, show/hide, enable/disable), the plan must explicitly cover both actions.
- Do not place likely rendering files (.html, .htm, .jinja, .j2, template files) in files_to_avoid unless evidence clearly shows they are irrelevant.
- Do not propose unrelated enhancements.
- Do not propose optional UX improvements, animations, notifications, refactors, cleanup, or extra features unless the task explicitly asks for them.
- Do not include orchestration metadata or repository-management steps.
- Prefer the smallest safe change set.
- STRICT LIMIT: Maximum 3 files_to_modify unless absolutely necessary.
- If you need more than 3 files, you MUST justify it in "risks".
- Return 2 to 5 concrete steps, not a long roadmap.
- Every schema field is required.
- Every file in files_to_modify must be justified in evidence.
- Every evidence item must use a real file path and a short reason.
- files_to_avoid may be [] if nothing should be avoided.
""".strip()

    print("\n[1/3] Asking coordinator...")
    _, coordinator_plan, coordinator_validation_problems = run_json_stage(
        client=client,
        role="coordinator",
        prompt=coordinator_prompt_base,
        expected_keys=[
            "task_type",
            "probable_cause",
            "files_to_modify",
            "files_to_avoid",
            "evidence",
            "steps",
            "risks",
        ],
        validator=validate_coordinator_plan,
        validation_context=real_files,
        task=task,
        label="Coordinator",
    )

    print_grounding_debug("Coordinator", coordinator_plan, used_context_files)

    coordinator_validation_problems.extend(context_grounding_problems(coordinator_plan, used_context_files))
    coordinator_validation_problems.extend(context_relevance_problems(coordinator_plan, used_context_files, used_context_score_map, used_context_match_count_map))
    coordinator_validation_problems.extend(unsupported_novelty_problems(coordinator_plan, task, used_context_files, relevant_context))
    coordinator_validation_problems = list(dict.fromkeys(coordinator_validation_problems))

    if coordinator_validation_problems:
        coordinator_repair_prompt = coordinator_prompt_base + f"""

Your previous coordinator attempt had validation problems:
{json.dumps(coordinator_validation_problems, ensure_ascii=False)}

Repair the plan.
Return one corrected JSON object only.
Do not repeat the same invalid paths or schema mistakes.
""".strip()

        print("\n[1/3] Coordinator repair pass...")
        _, repaired_coordinator_plan, repaired_coordinator_problems = run_json_stage(
            client=client,
            role="coordinator",
            prompt=coordinator_repair_prompt,
            expected_keys=[
                "task_type",
                "probable_cause",
                "files_to_modify",
                "files_to_avoid",
                "evidence",
                "steps",
                "risks",
            ],
            validator=validate_coordinator_plan,
            validation_context=real_files,
            task=task,
            label="Coordinator repair",
        )

        repaired_coordinator_problems.extend(context_grounding_problems(repaired_coordinator_plan, used_context_files))
        repaired_coordinator_problems.extend(context_relevance_problems(repaired_coordinator_plan, used_context_files, used_context_score_map, used_context_match_count_map))
        repaired_coordinator_problems.extend(unsupported_novelty_problems(repaired_coordinator_plan, task, used_context_files, relevant_context))
        repaired_coordinator_problems = list(dict.fromkeys(repaired_coordinator_problems))

        if len(repaired_coordinator_problems) <= len(coordinator_validation_problems):
            coordinator_plan = repaired_coordinator_plan
            coordinator_validation_problems = repaired_coordinator_problems
    supervisor_prompt_base = build_supervisor_prompt(
        task=task,
        task_kw=task_kw,
        real_files=real_files,
        coordinator_plan=coordinator_plan,
        coordinator_validation_problems=coordinator_validation_problems,
        used_context_files_text=used_context_files_text,
    )

    print("\n[2/3] Asking supervisor...")
    _, supervisor_plan, supervisor_validation_problems = run_json_stage(
        client=client,
        role="supervisor",
        prompt=supervisor_prompt_base,
        expected_keys=[
            "approved",
            "worker",
            "reason",
            "files_to_modify",
            "files_to_avoid",
            "evidence",
            "constraints",
            "corrected_steps",
        ],
        validator=validate_supervisor_plan,
        validation_context=real_files,
        task=task,
        label="Supervisor",
    )

    print_grounding_debug("Supervisor", supervisor_plan, used_context_files)

    supervisor_validation_problems.extend(context_grounding_problems(supervisor_plan, used_context_files))
    supervisor_validation_problems.extend(context_relevance_problems(supervisor_plan, used_context_files, used_context_score_map, used_context_match_count_map))
    supervisor_validation_problems.extend(unsupported_novelty_problems(supervisor_plan, task, used_context_files, relevant_context))
    supervisor_validation_problems = list(dict.fromkeys(supervisor_validation_problems))

    if supervisor_validation_problems:
        supervisor_repair_prompt = supervisor_prompt_base + f"""

Your previous supervisor attempt had validation problems:
{json.dumps(supervisor_validation_problems, ensure_ascii=False)}

Repair the plan.
Return one corrected JSON object only.
Do not repeat the same invalid paths or schema mistakes.
""".strip()

        print("\n[2/3] Supervisor repair pass...")
        _, repaired_supervisor_plan, repaired_supervisor_problems = run_json_stage(
            client=client,
            role="supervisor",
            prompt=supervisor_repair_prompt,
            expected_keys=[
                "approved",
                "worker",
                "reason",
                "files_to_modify",
                "files_to_avoid",
                "evidence",
                "constraints",
                "corrected_steps",
            ],
            validator=validate_supervisor_plan,
            validation_context=real_files,
            task=task,
            label="Supervisor repair",
        )

        repaired_supervisor_problems.extend(context_grounding_problems(repaired_supervisor_plan, used_context_files))
        repaired_supervisor_problems.extend(context_relevance_problems(repaired_supervisor_plan, used_context_files, used_context_score_map, used_context_match_count_map))
        repaired_supervisor_problems.extend(unsupported_novelty_problems(repaired_supervisor_plan, task, used_context_files, relevant_context))
        repaired_supervisor_problems = list(dict.fromkeys(repaired_supervisor_problems))

        if len(repaired_supervisor_problems) <= len(supervisor_validation_problems):
            supervisor_plan = repaired_supervisor_plan
            supervisor_validation_problems = repaired_supervisor_problems

    if supervisor_validation_problems:
        print("\n=== Final supervisor gating problems ===")
        for p in supervisor_validation_problems:
            print("-", p)
        print("Stopping before worker routing.")
        return

    approved = bool(supervisor_plan.get("approved", False))
    if not approved:
        print("\n[2/3] Coordinator revision from supervisor feedback...")

        coordinator_revision_prompt = coordinator_prompt_base + f"""

Supervisor rejected the previous plan.

Supervisor reason:
{str(supervisor_plan.get("reason", "")).strip()}

Supervisor corrected steps:
{json.dumps(normalize_list(supervisor_plan.get("corrected_steps", [])), ensure_ascii=False)}

Supervisor constraints:
{json.dumps(normalize_list(supervisor_plan.get("constraints", [])), ensure_ascii=False)}

Revise the coordinator plan so it addresses the supervisor rejection.
Return one corrected JSON object only.
Do not invent unsupported mechanisms or new architecture unless it is grounded in the provided context.
""".strip()

        _, revised_coordinator_plan, revised_coordinator_problems = run_json_stage(
            client=client,
            role="coordinator",
            prompt=coordinator_revision_prompt,
            expected_keys=[
                "task_type",
                "probable_cause",
                "files_to_modify",
                "files_to_avoid",
                "evidence",
                "steps",
                "risks",
            ],
            validator=validate_coordinator_plan,
            validation_context=real_files,
            task=task,
            label="Coordinator revision",
        )

        print_grounding_debug("Coordinator revision", revised_coordinator_plan, used_context_files)

        revised_coordinator_problems.extend(context_grounding_problems(revised_coordinator_plan, used_context_files))
        revised_coordinator_problems.extend(context_relevance_problems(revised_coordinator_plan, used_context_files, used_context_score_map, used_context_match_count_map))
        revised_coordinator_problems.extend(unsupported_novelty_problems(revised_coordinator_plan, task, used_context_files, relevant_context))
        revised_coordinator_problems = list(dict.fromkeys(revised_coordinator_problems))

        coordinator_plan = revised_coordinator_plan
        coordinator_validation_problems = revised_coordinator_problems

        supervisor_prompt_base = build_supervisor_prompt(
            task=task,
            task_kw=task_kw,
            real_files=real_files,
            coordinator_plan=coordinator_plan,
            coordinator_validation_problems=coordinator_validation_problems,
            used_context_files_text=used_context_files_text,
        )

        print("\n[2/3] Supervisor re-review...")
        _, supervisor_plan, supervisor_validation_problems = run_json_stage(
            client=client,
            role="supervisor",
            prompt=supervisor_prompt_base,
            expected_keys=[
                "approved",
                "worker",
                "reason",
                "files_to_modify",
                "files_to_avoid",
                "evidence",
                "constraints",
                "corrected_steps",
            ],
            validator=validate_supervisor_plan,
            validation_context=real_files,
            task=task,
            label="Supervisor re-review",
        )

        print_grounding_debug("Supervisor re-review", supervisor_plan, used_context_files)

        supervisor_validation_problems.extend(context_grounding_problems(supervisor_plan, used_context_files))
        supervisor_validation_problems.extend(context_relevance_problems(supervisor_plan, used_context_files, used_context_score_map, used_context_match_count_map))
        supervisor_validation_problems.extend(unsupported_novelty_problems(supervisor_plan, task, used_context_files, relevant_context))
        supervisor_validation_problems = list(dict.fromkeys(supervisor_validation_problems))

        approved = bool(supervisor_plan.get("approved", False))
        if supervisor_validation_problems or not approved:
            print("Stopping before worker routing.")
            return

    constraints = normalize_list(supervisor_plan.get("constraints", []))
    worker_type = str(supervisor_plan.get("worker", "")).strip().lower()

    if worker_type == "code":
        selected_worker_role = "code_worker"
    elif worker_type == "ui":
        selected_worker_role = "ui_worker"
    else:
        print("\nUnknown worker type returned by supervisor.")
        print(f"worker={worker_type!r}")
        return

    approved_files = normalize_list(supervisor_plan.get("files_to_modify", []))
    grounded_worker_files = [f for f in approved_files if f in used_context_files]
    out_of_context_approved_files = [f for f in approved_files if f not in used_context_files]

    if out_of_context_approved_files:
        print("\n=== Worker grounding gate ===")
        print("Approved files outside analyzed context:")
        for f in out_of_context_approved_files:
            print("-", f)

        if strict_worker_context_grounding:
            print("Stopping before worker routing because strict_worker_context_grounding is enabled.")
            return

    effective_worker_files = grounded_worker_files if strict_worker_context_grounding else approved_files

    if not effective_worker_files:
        print("\nNo grounded worker files available.")
        print("Stopping before worker routing.")
        return

    worker_request = {
        "task": task,
        "worker": worker_type,
        "selected_worker_role": selected_worker_role,
        "files_to_modify": effective_worker_files,
        "files_to_avoid": normalize_list(supervisor_plan.get("files_to_avoid", [])),
        "constraints": constraints,
        "steps": normalize_list(supervisor_plan.get("corrected_steps", [])),
        "evidence": supervisor_plan.get("evidence", []),
    }

    print("\n=== Worker routing ===")
    print(f"Selected worker type: {worker_type}")
    print(f"Selected worker role: {selected_worker_role}")

    print("\n=== Worker request preview ===")
    print(json.dumps(worker_request, indent=2, ensure_ascii=False))

    worker_context_blocks = []
    for rel_path in effective_worker_files:
        abs_path = root / rel_path
        if abs_path.exists() and abs_path.is_file():
            worker_content = read_file_safe(abs_path, max_chars=max_file_chars)
            worker_context_blocks.append(f"=== FILE: {rel_path} ===\n{worker_content}\n")
    worker_context = "\n".join(worker_context_blocks)

    worker_prompt_base = f"""
You are a specialized JSON generator acting as the implementation worker in a multi-agent coding system.
You are NOT a chat assistant.
Do not use chain-of-thought.
Do not write "Thinking..." or explanations.
Start your response with {{ and end with }}.

Your job is to translate the approved plan into one concise Aider-ready implementation task.

PRIMARY TASK TO SOLVE:
{task}

TASK KEYWORDS:
{json.dumps(task_kw, ensure_ascii=False)}

You must translate the approved plan for the PRIMARY TASK above exactly as written.

Approved worker request:
{json.dumps(worker_request, ensure_ascii=False)}

Approved file contents for this worker only:
{worker_context}

CRITICAL REMINDER BEFORE YOU ANSWER:
PRIMARY TASK: {task}
KEYWORDS: {json.dumps(task_kw, ensure_ascii=False)}
Output ONLY JSON. Do not write "Thinking..." or explanations.

Return exactly one valid JSON object with this exact schema:
{{
  "implementation_summary": "short text",
  "files_to_modify": ["real/project/file1", "real/project/file2"],
  "evidence": [{{"file": "real/project/file1", "reason": "why this file is relevant"}}],
  "aider_task": "concise implementation instruction for Aider"
}}

Hard rules:
- Output JSON only.
- Do not output markdown.
- Do not output analysis or thinking.
- Use ONLY files from this grounded approved list: {json.dumps(effective_worker_files, ensure_ascii=False)}
- Do not add any new files.
- Do not add unrelated improvements.
- Do not mention orchestration, path normalization, repo management, or file path reference changes.
- Keep the aider_task concise, direct, and implementation-focused.
- The aider_task must preserve every listed constraint.
- The aider_task must incorporate the practical reasoning from evidence so the implementation tool understands why each chosen file matters.
- Do not drop the file-specific rationale from evidence when writing aider_task.
""".strip()

    print(f"\n[3/3] Asking {selected_worker_role}...")
    _, worker_plan, worker_validation_problems = run_json_stage(
        client=client,
        role=selected_worker_role,
        prompt=worker_prompt_base,
        expected_keys=[
            "implementation_summary",
            "files_to_modify",
            "evidence",
            "aider_task",
        ],
        validator=validate_worker_plan,
        validation_context=effective_worker_files,
        task=task,
        label="Worker",
    )

    if worker_validation_problems:
        worker_repair_prompt = worker_prompt_base + f"""

Your previous worker attempt had validation problems:
{json.dumps(worker_validation_problems, ensure_ascii=False)}

Repair the plan.
Return one corrected JSON object only.
Do not add files outside the approved list.
""".strip()

        print(f"\n[3/3] {selected_worker_role} repair pass...")
        _, repaired_worker_plan, repaired_worker_problems = run_json_stage(
            client=client,
            role=selected_worker_role,
            prompt=worker_repair_prompt,
            expected_keys=[
                "implementation_summary",
                "files_to_modify",
                "evidence",
                "aider_task",
            ],
            validator=validate_worker_plan,
            validation_context=effective_worker_files,
            task=task,
            label="Worker repair",
        )

        if len(repaired_worker_problems) <= len(worker_validation_problems):
            worker_plan = repaired_worker_plan
            worker_validation_problems = repaired_worker_problems

    if worker_validation_problems:
        print("Stopping before Aider handoff.")
        return

    aider_task = str(worker_plan.get("aider_task", "")).strip()

    print("\n=== Aider task preview ===")
    print(aider_task)

    run_payload = {
        "project_path": str(root),
        "task": task,
        "used_context_files": used_context_files,
        "total_context_chars": total_context_chars,
        "coordinator_plan": coordinator_plan,
        "coordinator_validation_problems": coordinator_validation_problems,
        "supervisor_plan": supervisor_plan,
        "supervisor_validation_problems": supervisor_validation_problems,
        "worker_request": worker_request,
        "approved_files": approved_files,
        "grounded_worker_files": grounded_worker_files,
        "out_of_context_approved_files": out_of_context_approved_files,
        "worker_plan": worker_plan,
        "worker_validation_problems": worker_validation_problems,
        "aider_task": aider_task,
    }

    run_file = runs_dir / "last_run.json"
    aider_task_file = runs_dir / "last_aider_task.txt"

    run_file.write_text(json.dumps(run_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    aider_task_file.write_text(aider_task, encoding="utf-8")

    print("\n=== Run artifact ===")
    print(run_file)

    print("\n=== Aider task artifact ===")
    print(aider_task_file)


if __name__ == "__main__":
    main()
