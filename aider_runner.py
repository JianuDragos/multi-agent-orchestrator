import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_FILE = REPO_ROOT / "runs" / "last_run.json"
DEFAULT_TASK_FILE = REPO_ROOT / "runs" / "last_aider_task.txt"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_paths(items):
    seen = set()
    out = []
    for item in items or []:
        value = str(item).strip().replace("\\", "/")
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def is_git_repo(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def resolve_edit_files(run_payload: dict):
    candidates = [
        run_payload.get("worker_plan", {}).get("files_to_modify", []),
        run_payload.get("grounded_worker_files", []),
        run_payload.get("approved_files", []),
    ]
    for candidate in candidates:
        files = unique_paths(candidate)
        if files:
            return files
    return []


def resolve_read_files(run_payload: dict, edit_files: list[str]):
    edit_set = set(edit_files)
    used_context_files = unique_paths(run_payload.get("used_context_files", []))
    return [path for path in used_context_files if path not in edit_set]


def derive_aider_model(config: dict, run_payload: dict) -> str:
    aider_cfg = config.get("aider", {})
    explicit = str(aider_cfg.get("model", "")).strip()
    if explicit:
        return explicit

    models = config.get("models", {})
    selected_role = str(
        run_payload.get("worker_request", {}).get("selected_worker_role", "")
    ).strip()

    if selected_role and selected_role in models:
        base_model = str(models[selected_role]).strip()
    else:
        worker_type = str(run_payload.get("worker_request", {}).get("worker", "")).strip().lower()
        if worker_type == "ui":
            base_model = str(models.get("ui_worker", "")).strip()
        else:
            base_model = str(models.get("code_worker", "")).strip()

    if not base_model:
        return ""

    if "/" in base_model:
        return base_model

    return f"ollama_chat/{base_model}"


def build_command(config: dict, run_payload: dict, prompt_file: Path, prompt_text: str):
    aider_cfg = config.get("aider", {})

    project_path = Path(run_payload.get("project_path", "")).resolve()
    if not project_path.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")

    edit_files = resolve_edit_files(run_payload)
    if not edit_files:
        raise ValueError("No files_to_modify found in run payload")

    read_files = resolve_read_files(run_payload, edit_files)

    aider_bin = shutil.which("aider")
    if not aider_bin:
        raise RuntimeError("aider executable not found in PATH")

    command = [aider_bin]

    model = derive_aider_model(config, run_payload)
    if model:
        command += ["--model", model]

    edit_format = str(aider_cfg.get("edit_format", "")).strip()
    if edit_format:
        command += ["--edit-format", edit_format]

    if bool(aider_cfg.get("yes_always", True)):
        command.append("--yes-always")

    if bool(aider_cfg.get("no_pretty", True)):
        command.append("--no-pretty")

    if bool(aider_cfg.get("read_context_files", True)):
        for path in read_files:
            command += ["--read", path]

    for path in edit_files:
        command += ["--file", path]

    if not is_git_repo(project_path):
        command.append("--no-git")

    extra_args = aider_cfg.get("extra_args", [])
    if isinstance(extra_args, list):
        command.extend(str(arg) for arg in extra_args if str(arg).strip())

    if prompt_file.exists():
        command += ["--message-file", str(prompt_file.resolve())]
    else:
        command += ["--message", prompt_text]

    command.append("--exit")

    return command, project_path, edit_files, read_files, model


def main():
    parser = argparse.ArgumentParser(description="Run Aider using the latest orchestrator artifacts.")
    parser.add_argument("--run-file", default=str(DEFAULT_RUN_FILE), help="Path to runs/last_run.json")
    parser.add_argument("--task-file", default=str(DEFAULT_TASK_FILE), help="Path to runs/last_aider_task.txt")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    args = parser.parse_args()

    run_file = Path(args.run_file).resolve()
    task_file = Path(args.task_file).resolve()

    if not run_file.exists():
        raise FileNotFoundError(f"Run artifact not found: {run_file}")

    run_payload = load_json(run_file)

    config_path = REPO_ROOT / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")

    config = load_json(config_path)

    prompt_text = ""
    if task_file.exists():
        prompt_text = task_file.read_text(encoding="utf-8").strip()
    else:
        prompt_text = str(run_payload.get("aider_task", "")).strip()

    if not prompt_text:
        raise ValueError("No Aider task found in task file or run payload")

    command, project_path, edit_files, read_files, model = build_command(
        config=config,
        run_payload=run_payload,
        prompt_file=task_file,
        prompt_text=prompt_text,
    )

    print("=== Aider runner ===")
    print("Project path:", project_path)
    print("Model:", model or "(aider default)")
    print("Edit files:", edit_files)
    print("Read-only files:", read_files)
    print("Run file:", run_file)
    print("Task file:", task_file if task_file.exists() else "(missing, using embedded aider_task)")
    print("\nCommand:")
    print(" ".join(subprocess.list2cmdline([part]) if " " in part else part for part in command))

    if args.dry_run:
        print("\nDry run only. No command executed.")
        return

    print("\nLaunching Aider...\n")
    result = subprocess.run(command, cwd=project_path)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
