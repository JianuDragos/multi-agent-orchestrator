import posixpath
import re


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "without",
    "fix", "update", "change", "modify", "make", "add", "remove", "improve", "issue",
    "bug", "problem", "button", "logic", "task", "file", "files"
}


def normalize_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def normalize_model_path(path):
    value = str(path or "").strip().replace("\\", "/")

    while "//" in value:
        value = value.replace("//", "/")

    if len(value) >= 2 and value[1] == ":":
        value = value[2:]

    if value.startswith("./"):
        value = value[2:]

    value = value.lstrip("/")

    normalized = posixpath.normpath(value)

    if normalized == ".":
        return ""

    return normalized


def normalize_plan_paths(plan):
    if not isinstance(plan, dict):
        return plan

    normalized = dict(plan)

    for field in ("files_to_modify", "files_to_avoid"):
        normalized[field] = [
            path for path in
            (normalize_model_path(x) for x in normalize_list(plan.get(field, [])))
            if path
        ]

    evidence = plan.get("evidence", [])
    if isinstance(evidence, list):
        normalized_evidence = []
        for item in evidence:
            if not isinstance(item, dict):
                normalized_evidence.append(item)
                continue
            normalized_item = dict(item)
            normalized_item["file"] = normalize_model_path(item.get("file", ""))
            normalized_evidence.append(normalized_item)
        normalized["evidence"] = normalized_evidence

    return normalized


def _validate_paths(paths, real_files, field_name):
    problems = []
    for path in normalize_list(paths):
        normalized = normalize_model_path(path)
        if not normalized:
            problems.append(f"{field_name} contains an empty path")
            continue
        if normalized not in real_files:
            problems.append(f"{field_name} contains non-existent path: {normalized}")
    return problems


def _validate_no_overlap(a, b, left_name, right_name):
    left = set(normalize_model_path(x) for x in normalize_list(a) if normalize_model_path(x))
    right = set(normalize_model_path(x) for x in normalize_list(b) if normalize_model_path(x))
    overlap = sorted(left & right)
    return [
        f"path appears in both {left_name} and {right_name}: {path}"
        for path in overlap
    ]


def _tokenize(text):
    words = re.findall(r"[a-zA-Z0-9_]+", str(text).lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def _task_alignment_problems(plan, task):
    if not task:
        return []

    task_tokens = set(_tokenize(task))
    if not task_tokens:
        return []

    probable_cause = str(plan.get("probable_cause", ""))
    reason = str(plan.get("reason", ""))
    steps = " ".join(normalize_list(plan.get("steps", [])))
    corrected_steps = " ".join(normalize_list(plan.get("corrected_steps", [])))
    files_to_modify = " ".join(normalize_list(plan.get("files_to_modify", [])))

    evidence_parts = []
    evidence = plan.get("evidence", [])
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                evidence_parts.append(str(item.get("file", "")))
                evidence_parts.append(str(item.get("reason", "")))

    plan_text = " ".join([
        probable_cause,
        reason,
        steps,
        corrected_steps,
        files_to_modify,
        " ".join(evidence_parts),
    ]).lower()

    matched = sorted(token for token in task_tokens if token in plan_text)
    match_ratio = len(matched) / max(len(task_tokens), 1)

    problems = []

    if len(matched) == 0:
        problems.append("plan appears off-task: no meaningful task keywords appear in the plan")
    elif match_ratio < 0.20:
        problems.append(
            f"plan may be off-task: weak task alignment "
            f"({len(matched)}/{len(task_tokens)} task keywords matched: {matched})"
        )

    return problems


def _validate_evidence(plan, real_files):
    problems = []
    evidence = plan.get("evidence", None)

    if evidence is None:
        return ['evidence field is missing']

    if not isinstance(evidence, list) or not evidence:
        return ['evidence must be a non-empty list']

    evidence_files = []

    for idx, item in enumerate(evidence):
        if not isinstance(item, dict):
            problems.append(f"evidence[{idx}] must be an object")
            continue

        file_path = normalize_model_path(item.get("file", ""))
        reason = str(item.get("reason", "")).strip()

        if not file_path:
            problems.append(f"evidence[{idx}].file is missing or empty")
        elif file_path not in real_files:
            problems.append(f"evidence[{idx}].file contains non-existent path: {file_path}")
        else:
            evidence_files.append(file_path)

        if not reason:
            problems.append(f"evidence[{idx}].reason is missing or empty")

    for f in normalize_list(plan.get("files_to_modify", [])):
        nf = normalize_model_path(f)
        if nf and nf not in evidence_files:
            problems.append(f"files_to_modify path is not justified in evidence: {nf}")

    return problems


def validate_coordinator_plan(plan, real_files, task=None):
    problems = []

    if not isinstance(plan, dict):
        return ["Coordinator plan is not a JSON object"]

    plan = normalize_plan_paths(plan)

    task_type = str(plan.get("task_type", "")).strip().lower()
    if task_type not in {"code", "ui", "mixed"}:
        problems.append(f"invalid task_type: {task_type!r}")

    probable_cause = str(plan.get("probable_cause", "")).strip()
    if not probable_cause:
        problems.append("probable_cause is missing or empty")

    files_to_modify = normalize_list(plan.get("files_to_modify", []))
    files_to_avoid = normalize_list(plan.get("files_to_avoid", []))
    steps = normalize_list(plan.get("steps", []))
    risks = normalize_list(plan.get("risks", []))

    if not files_to_modify:
        problems.append("files_to_modify is missing or empty")
    if not steps:
        problems.append("steps is missing or empty")
    if not risks:
        problems.append("risks is missing or empty")

    problems.extend(_validate_paths(files_to_modify, real_files, "files_to_modify"))
    problems.extend(_validate_paths(files_to_avoid, real_files, "files_to_avoid"))
    problems.extend(_validate_no_overlap(files_to_modify, files_to_avoid, "files_to_modify", "files_to_avoid"))
    problems.extend(_validate_evidence(plan, real_files))
    problems.extend(_task_alignment_problems(plan, task))

    return problems


def validate_supervisor_plan(plan, real_files, task=None):
    problems = []

    if not isinstance(plan, dict):
        return ["Supervisor plan is not a JSON object"]

    plan = normalize_plan_paths(plan)

    approved = plan.get("approved", None)
    if not isinstance(approved, bool):
        problems.append("approved must be a boolean")

    worker = str(plan.get("worker", "")).strip().lower()
    if worker not in {"code", "ui"}:
        problems.append(f"invalid worker: {worker!r}")

    reason = str(plan.get("reason", "")).strip()
    if not reason:
        problems.append("reason is missing or empty")

    files_to_modify = normalize_list(plan.get("files_to_modify", []))
    files_to_avoid = normalize_list(plan.get("files_to_avoid", []))
    corrected_steps = normalize_list(plan.get("corrected_steps", []))

    if not files_to_modify:
        problems.append("files_to_modify is missing or empty")
    if not corrected_steps:
        problems.append("corrected_steps is missing or empty")
    if "constraints" not in plan:
        problems.append('constraints field is missing')

    problems.extend(_validate_paths(files_to_modify, real_files, "files_to_modify"))
    problems.extend(_validate_paths(files_to_avoid, real_files, "files_to_avoid"))
    problems.extend(_validate_no_overlap(files_to_modify, files_to_avoid, "files_to_modify", "files_to_avoid"))
    problems.extend(_validate_evidence(plan, real_files))
    problems.extend(_task_alignment_problems(plan, task))

    return problems
