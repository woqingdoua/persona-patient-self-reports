#!/usr/bin/env python3
"""Offline release hygiene checks. Never imports research workflows."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__"}
ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".cff", ".json"}
SECRET = re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}")
PRIVATE_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/")
DATA_SUFFIXES = {
    ".csv", ".tsv", ".gz", ".xlsx", ".xls", ".pdf", ".docx", ".bak",
    ".npy", ".npz", ".pt", ".pth", ".safetensors", ".bin", ".zip",
}


def check(root=ROOT):
    errors = []
    sources = 0
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
            continue
        files.append(path)
        if path.is_symlink():
            errors.append(f"{rel}: symlink not allowed in release")
            continue
        if any(suffix.lower() in DATA_SUFFIXES for suffix in path.suffixes):
            errors.append(f"{rel}: data/binary artifact in code-only release")
            continue
        if path.name == ".env" or path.suffix in {".pem", ".key"}:
            errors.append(f"{rel}: sensitive configuration file")
            continue
        if path.name != ".gitignore" and path.suffix not in ALLOWED_SUFFIXES:
            errors.append(f"{rel}: unexpected file type")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            errors.append(f"{rel}: not UTF-8 text")
            continue
        if SECRET.search(text):
            errors.append(f"{rel}: potential embedded credential (value suppressed)")
        if PRIVATE_PATH.search(text):
            errors.append(f"{rel}: private absolute path")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(rel))
                compile(text, str(rel), "exec")
                sources += 1
            except SyntaxError as exc:
                errors.append(f"{rel}:{exc.lineno}: syntax error")
        if path.suffix == ".md":
            for link in re.findall(r"\[[^\]]+\]\(([^)\s]+)\)", text):
                if "://" in link or link.startswith(("#", "mailto:")):
                    continue
                target = link.split("#", 1)[0]
                if target and not (path.parent / target).exists():
                    errors.append(f"{rel}: missing local link {target}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"PASS: {sources} Python files compile; {len(files)} release files checked.")
    print("No blocked artifact types, matching credential/private-path patterns, or broken local Markdown links.")
    print("Scope: static hygiene only, not a clinical-data or API reproducibility test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check())
