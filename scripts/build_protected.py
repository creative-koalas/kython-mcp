#!/usr/bin/env python3
"""Build a Cython-protected deployment artifact.

The normal source tree remains unchanged. The generated artifact is intended
for customer/B-side deployment where core Python modules should be shipped as
native extension modules instead of plaintext source.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import importlib.machinery
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
import tokenize
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - this project requires py3.11+
    tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = Path("scripts/build_protected.py")
MANIFEST_FILENAME = "PROTECTED_ARTIFACT_MANIFEST.json"

DEFAULT_RUNTIME_INCLUDE_GLOBS = (
    "alembic.ini",
    "alembic/*.mako",
    "alembic/**/*.mako",
    "prisma/schema.prisma",
)

DEFAULT_PLAINTEXT_RUNTIME_PY_GLOBS = (
    "alembic/env.py",
    "alembic/versions/*.py",
    "alembic/versions/**/*.py",
    "scripts/bootstrap.py",
    "scripts/run_server.py",
)

DEFAULT_RUNTIME_EXCLUDE_GLOBS = (
    "*.md",
    "AGENTS.md",
    "CLAUDE.md",
    "REBALANCE_DESIGN.md",
    "Dockerfile*",
    "uv.lock",
    ".github/**",
    ".pytest_keys/**",
    "config/**",
    "docs/**",
    "k8s/**",
    "tests/**",
    "*.bug.log",
)

SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".serena",
    ".tox",
    ".uv",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "dist-protected",
    "htmlcov",
    "node_modules",
    "tests",
    "venv",
}

SKIP_FILE_NAMES = {
    ".coverage",
    ".DS_Store",
}

SKIP_FILE_PREFIXES = (
    ".coverage.",
    ".env",
)

FORBIDDEN_ARTIFACT_SUFFIXES = {".c", ".pyc", ".pyx"}
TEXT_AUDIT_SUFFIXES = {".ini", ".json", ".prisma", ".py", ".toml"}
NATIVE_AUDIT_SUFFIXES = (".dylib", ".pyd", ".so")

KNOWN_SECRET_PATTERNS = (
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("github token", re.compile(r"\bgh[pousr]_[0-9A-Za-z_]{30,}\b")),
    ("openai api key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b")),
)

SECRET_ASSIGNMENT_RE = re.compile(
    r"""
    ["']?
    \b
    (?:api[_-]?key|secret(?:[_-]?key)?|client[_-]?secret|access[_-]?token|
       refresh[_-]?token|auth[_-]?token|bearer[_-]?token|private[_-]?key|
       token|jwt|key|password|passwd|pwd)
    \b
    ["']?
    \s*[:=]\s*
    ["']?
    ([^\s"',;#}\])]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

WEAK_PASSWORD_ASSIGNMENT_RE = re.compile(
    r"""
    ["']?
    \b(?:password|passwd|pwd)\b
    ["']?
    \s*[:=]\s*
    ["']?
    (admin|changeme|default|dev|password|postgres|root|secret|test|123456)
    ["']?
    """,
    re.IGNORECASE | re.VERBOSE,
)

WEAK_PASSWORD_FALLBACK_RE = re.compile(
    r"""
    \b(?:password|passwd|pwd|pg_password)\b
    .*
    ["'](admin|changeme|default|dev|password|postgres|root|secret|test|123456)["']
    """,
    re.IGNORECASE | re.VERBOSE,
)

INTERNAL_URL_RE = re.compile(
    r"""
    https?://
    (?:
        localhost|
        127(?:\.\d{1,3}){3}|
        10(?:\.\d{1,3}){3}|
        172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|
        192\.168(?:\.\d{1,3}){2}|
        [A-Za-z0-9.-]*(?:\.internal|\.local|\.svc|svc\.cluster\.local|internal|intranet|corp)[A-Za-z0-9.-]*
    )
    (?::\d+)?
    (?:/[^\s"'<>)]*)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

SLIMMING_RISK_MARKERS = (
    "FastAPI",
    "FastMCP",
    "Field(",
    "BaseModel",
    "__doc__",
    "@app.",
    "@mcp.",
    "@router.",
    "APIRouter",
    "Body(",
    "fastapi",
    "fastmcp",
    "Header(",
    "inspect.",
    "pydantic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Cython-protected deployment tree and tarball.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dist-protected/app",
        help="Protected deployment tree path, relative to the repo root by default.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "Python file or directory to include in the protected runtime. "
            "May be repeated. Defaults are discovered from pyproject.toml plus runtime scripts."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate target discovery and print Cython build inputs without creating output.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing protected output tree without rebuilding it.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run the sensitive-string audit against an existing protected output tree.",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help=(
            "Build a production delivery tree: compile, strip, verify, audit, "
            "and remove the local manifest before archiving."
        ),
    )
    parser.add_argument(
        "--allow-plaintext-py",
        action="store_true",
        help="Allow runtime .py files outside the configured plaintext allowlist.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip creating the protected .tar.gz archive.",
    )
    parser.add_argument(
        "--no-strip",
        action="store_true",
        help="Skip stripping native extension symbols after Cython compilation.",
    )
    args = parser.parse_args()
    if args.production and args.no_strip:
        parser.error("--production requires stripped native extensions; remove --no-strip.")
    if args.production and (args.check_only or args.verify_only or args.audit_only):
        parser.error("--production is a build mode and cannot be combined with check/verify/audit-only.")
    return args


def load_pyproject() -> dict:
    if tomllib is None:
        raise SystemExit("Python 3.11+ is required to read pyproject.toml")
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return {}
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


def project_name(pyproject: dict) -> str:
    return (
        pyproject.get("project", {}).get("name")
        or REPO_ROOT.name
    ).replace("/", "-")


def project_version(pyproject: dict) -> str:
    return pyproject.get("project", {}).get("version") or "0.0.0"


def has_python_files(path: Path) -> bool:
    if path.is_file():
        return path.suffix == ".py"
    return any(p.suffix == ".py" for p in path.rglob("*.py") if not should_skip(p.relative_to(REPO_ROOT)))


def should_skip(rel_path: Path) -> bool:
    parts = rel_path.parts
    if any(part in SKIP_DIR_NAMES for part in parts[:-1]):
        return True
    if rel_path.name in SKIP_FILE_NAMES:
        return True
    if any(rel_path.name.startswith(prefix) for prefix in SKIP_FILE_PREFIXES):
        return True
    return False


def split_patterns(value: str) -> list[str]:
    return [
        item.strip()
        for chunk in value.split(os.pathsep)
        for item in chunk.split(",")
        if item.strip()
    ]


def protected_tool(pyproject: dict) -> dict:
    value = pyproject.get("tool", {}).get("protected-artifacts", {})
    return value if isinstance(value, dict) else {}


def configured_runtime_globs(pyproject: dict, key: str, env_name: str) -> list[str]:
    configured = protected_tool(pyproject).get(key, [])
    if isinstance(configured, str):
        patterns = [configured]
    else:
        patterns = [item for item in configured if isinstance(item, str)]
    patterns.extend(split_patterns(os.getenv(env_name, "")))
    return patterns


def runtime_include_globs(pyproject: dict) -> list[str]:
    return [
        *DEFAULT_RUNTIME_INCLUDE_GLOBS,
        *configured_runtime_globs(
            pyproject,
            "include-files",
            "PROTECTED_ARTIFACT_INCLUDE_FILES",
        ),
    ]


def runtime_exclude_globs(pyproject: dict) -> list[str]:
    return [
        *DEFAULT_RUNTIME_EXCLUDE_GLOBS,
        *configured_runtime_globs(
            pyproject,
            "exclude-files",
            "PROTECTED_ARTIFACT_EXCLUDE_FILES",
        ),
    ]


def plaintext_runtime_py_globs(pyproject: dict) -> list[str]:
    return [
        *DEFAULT_PLAINTEXT_RUNTIME_PY_GLOBS,
        *configured_runtime_globs(
            pyproject,
            "plaintext-runtime-py",
            "PROTECTED_PLAINTEXT_RUNTIME_PY",
        ),
    ]


def sensitive_audit_allowlist(pyproject: dict) -> list[re.Pattern[str]]:
    configured = protected_tool(pyproject).get("sensitive-audit-allowlist", [])
    if isinstance(configured, str):
        patterns = [configured]
    else:
        patterns = [item for item in configured if isinstance(item, str)]
    patterns.extend(split_patterns(os.getenv("PROTECTED_SENSITIVE_AUDIT_ALLOWLIST", "")))

    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise SystemExit(f"Invalid sensitive audit allowlist pattern {pattern!r}: {exc}") from exc
    return compiled


def matches_glob(rel_path: Path, patterns: list[str] | tuple[str, ...]) -> bool:
    value = rel_path.as_posix()
    return any(
        fnmatch.fnmatchcase(value, pattern)
        or fnmatch.fnmatchcase(rel_path.name, pattern)
        for pattern in patterns
    )


def add_target(targets: list[Path], rel_path: str | Path) -> None:
    rel = Path(rel_path)
    path = (REPO_ROOT / rel).resolve()
    if not path.exists() or not has_python_files(path):
        return
    for existing in list(targets):
        try:
            if path.is_relative_to(existing):
                return
            if existing.is_relative_to(path):
                targets.remove(existing)
        except ValueError:
            pass
    targets.append(path)


def discover_targets(pyproject: dict, explicit_targets: list[str]) -> list[Path]:
    targets: list[Path] = []

    env_targets = os.getenv("PROTECTED_ARTIFACT_TARGETS", "")
    configured_targets = explicit_targets or [
        item.strip()
        for chunk in env_targets.split(os.pathsep)
        for item in chunk.split(",")
        if item.strip()
    ]
    if configured_targets:
        for item in configured_targets:
            add_target(targets, item)
        return sorted(targets)

    wheel = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
    )
    for package in wheel.get("packages", []) if isinstance(wheel, dict) else []:
        add_target(targets, package)

    if not targets and (REPO_ROOT / "src").exists():
        add_target(targets, "src")

    scripts_dir = REPO_ROOT / "scripts"
    if scripts_dir.exists():
        for script in sorted(scripts_dir.rglob("*.py")):
            rel = script.relative_to(REPO_ROOT)
            if rel == BUILD_SCRIPT:
                continue
            add_target(targets, rel)

    if (REPO_ROOT / "alembic").exists():
        add_target(targets, "alembic")

    for script in sorted(REPO_ROOT.glob("*.py")):
        add_target(targets, script.relative_to(REPO_ROOT))

    return sorted(targets)


def configured_cython_targets(pyproject: dict) -> list[Path]:
    configured: list[str] = []
    configured.extend(split_patterns(os.getenv("PROTECTED_CYTHON_TARGETS", "")))

    for item in protected_tool(pyproject).get("cython-targets", []):
        if isinstance(item, str) and item.strip():
            configured.append(item.strip())

    targets: list[Path] = []
    for item in configured:
        path = (REPO_ROOT / item).resolve()
        if not path.exists() or not path.is_file() or path.suffix != ".py":
            raise SystemExit(f"Cython target must be an existing Python file: {item}")
        targets.append(path)
    return sorted(set(targets))


def plaintext_runtime_py_files(pyproject: dict) -> list[Path]:
    files: list[Path] = []
    for pattern in plaintext_runtime_py_globs(pyproject):
        for path in REPO_ROOT.glob(pattern):
            if not path.is_file() or path.suffix != ".py":
                continue
            rel = relative(path)
            if should_skip(rel) or rel == BUILD_SCRIPT:
                continue
            files.append(path.resolve())
    return sorted(set(files))


def relative(path: Path) -> Path:
    return path.relative_to(REPO_ROOT)


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def runtime_python_files(targets: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for target in targets:
        if target.is_file():
            if target.suffix == ".py" and relative(target) != BUILD_SCRIPT and not should_skip(relative(target)):
                files.add(target.resolve())
            continue
        for source in target.rglob("*.py"):
            rel = relative(source)
            if rel == BUILD_SCRIPT or should_skip(rel):
                continue
            files.add(source.resolve())
    return sorted(files)


def assert_no_unprotected_runtime_python(
    targets: list[Path],
    allow_plaintext: bool,
    cython_targets: list[Path],
    plaintext_runtime_python_files: list[Path],
) -> None:
    if allow_plaintext:
        return
    allowed = {path.resolve() for path in cython_targets} | {
        path.resolve() for path in plaintext_runtime_python_files
    }
    unprotected = [
        str(relative(source))
        for source in runtime_python_files(targets)
        if source.resolve() not in allowed
    ]
    if unprotected:
        sample = "\n  ".join(unprotected[:30])
        raise SystemExit(
            "Refusing to build: runtime Python files are neither Cython targets nor allowed plaintext files:\n"
            f"  {sample}\n"
            "Add them to [tool.protected-artifacts].cython-targets or plaintext-runtime-py."
        )


def clean_output(output_dir: Path) -> None:
    if output_dir == REPO_ROOT or REPO_ROOT not in output_dir.parents:
        raise SystemExit("Output directory must be a subdirectory of this repository")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_runtime_files(
    output_dir: Path,
    targets: list[Path],
    include_globs: list[str],
    exclude_globs: list[str],
    plaintext_runtime_python_files: list[Path] | None = None,
) -> list[str]:
    plaintext_runtime_python_files = plaintext_runtime_python_files or []
    plaintext_runtime_python_set = {path.resolve() for path in plaintext_runtime_python_files}
    copied: list[str] = []
    for source in sorted(REPO_ROOT.rglob("*")):
        if source.is_dir():
            continue
        rel = relative(source)
        if rel == BUILD_SCRIPT or should_skip(rel):
            continue
        if matches_glob(rel, exclude_globs):
            continue
        is_plaintext_runtime_py = source.resolve() in plaintext_runtime_python_set
        if rel.suffix == ".py" and not is_plaintext_runtime_py:
            continue
        in_runtime_target = any(source == target or is_under(source, target) for target in targets)
        explicitly_included = is_plaintext_runtime_py or matches_glob(rel, include_globs)
        if not in_runtime_target and not explicitly_included:
            continue
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied.append(str(rel))
    return copied


def should_slim_plaintext_python(rel: str, content: str) -> bool:
    lowered = rel.lower()
    risky_names = ("server.py", "schemas.py", "models.py", "routes.py", "api.py")
    if lowered.endswith(risky_names) or "/server.py" in lowered:
        return False
    return not any(marker in content for marker in SLIMMING_RISK_MARKERS)


def docstring_statement_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Module)):
            continue
        body = getattr(node, "body", None)
        if not body or len(body) < 2:
            continue
        first = body[0]
        if not isinstance(first, ast.Expr):
            continue
        value = first.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        start = getattr(first, "lineno", None)
        end = getattr(first, "end_lineno", start)
        if start is not None and end is not None:
            ranges.append((start, end))
    return ranges


def remove_plain_docstrings(content: str) -> str:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content
    ranges = docstring_statement_ranges(tree)
    if not ranges:
        return content

    lines = content.splitlines(keepends=True)
    for start, end in sorted(ranges, reverse=True):
        for index in range(start - 1, min(end, len(lines))):
            lines[index] = "\n" if lines[index].endswith("\n") else ""
    candidate = "".join(lines)
    try:
        compile(candidate, "<protected-slimming>", "exec")
    except SyntaxError:
        return content
    return candidate


def remove_python_comments(content: str) -> str:
    tokens: list[tokenize.TokenInfo] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(content).readline):
            if token.type == tokenize.COMMENT:
                line_no = token.start[0]
                keep_shebang = line_no == 1 and token.string.startswith("#!")
                keep_encoding = line_no <= 2 and "coding" in token.string
                if not keep_shebang and not keep_encoding:
                    continue
            tokens.append(token)
        return tokenize.untokenize(tokens)
    except tokenize.TokenError:
        return content


def slim_python_source(content: str) -> str:
    candidate = remove_python_comments(remove_plain_docstrings(content))
    try:
        compile(candidate, "<protected-slimming>", "exec")
    except SyntaxError:
        return content
    return candidate


def slim_plaintext_runtime_python(output_dir: Path, plaintext_files: list[str]) -> list[str]:
    slimmed: list[str] = []
    for rel in plaintext_files:
        source = output_dir / rel
        if not source.exists() or source.suffix != ".py":
            continue
        content = source.read_text(encoding="utf-8", errors="ignore")
        if not should_slim_plaintext_python(rel, content):
            continue
        candidate = slim_python_source(content)
        if candidate == content:
            continue
        source.write_text(candidate, encoding="utf-8")
        slimmed.append(rel)
    return slimmed


def run_command(command: list[str]) -> str:
    display = " ".join(command)
    print(f"+ {display}")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout.strip():
        print(completed.stdout.rstrip())
    return completed.stdout


def hatch_wheel_packages(pyproject: dict) -> list[Path]:
    wheel = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
    )
    packages = wheel.get("packages", []) if isinstance(wheel, dict) else []
    return [Path(item) for item in packages if isinstance(item, str) and item]


def strip_src_prefix_for_imports(pyproject: dict) -> bool:
    packages = hatch_wheel_packages(pyproject)
    if any(package.parts == ("src",) for package in packages):
        return False
    return (REPO_ROOT / "src").exists()


def cython_module_name(target: Path, pyproject: dict | None = None) -> str:
    pyproject = pyproject or {}
    rel = relative(target.with_suffix(""))
    parts = rel.parts
    if parts and parts[0] == "src" and strip_src_prefix_for_imports(pyproject):
        parts = parts[1:]
    return ".".join(parts)


def cython_build_lib_root(output_dir: Path, cython_targets: list[Path], pyproject: dict) -> Path:
    if strip_src_prefix_for_imports(pyproject) and any(relative(target).parts[:1] == ("src",) for target in cython_targets):
        return output_dir / "src"
    return output_dir


def find_cython_extension(output_dir: Path, target: Path) -> Path | None:
    dest_dir = output_dir / relative(target).parent
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = dest_dir / f"{target.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def assert_cython_extensions_exist(output_dir: Path, cython_targets: list[Path]) -> list[str]:
    compiled: list[str] = []
    missing: list[str] = []
    for target in cython_targets:
        extension = find_cython_extension(output_dir, target)
        if extension is None:
            missing.append(str(relative(target)))
        else:
            compiled.append(str(extension.relative_to(output_dir)))
    if missing:
        sample = "\n  ".join(missing[:20])
        raise SystemExit(
            "Protected output is missing Cython extension files for:\n"
            f"  {sample}"
        )
    return compiled


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    if len(node.test.ops) != 1 or not isinstance(node.test.ops[0], ast.Eq):
        return False
    if len(node.test.comparators) != 1:
        return False

    left = node.test.left
    right = node.test.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def has_top_level_main_function(tree: ast.Module) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
        for node in tree.body
    )


def rewrite_main_guard_as_main_function(source_text: str) -> str:
    """Expose top-level ``if __name__ == "__main__"`` blocks as ``main()``.

    Python cannot execute C extension modules with ``python -m`` because the
    extension loader has no code object. This build-time rewrite keeps the
    source tree unchanged while making compiled entrypoint modules callable via
    ``from package.module import main; main()``.
    """

    tree = ast.parse(source_text)
    if has_top_level_main_function(tree):
        return source_text

    main_guards = [node for node in tree.body if is_main_guard(node)]
    if not main_guards:
        return source_text

    guard = main_guards[-1]
    if guard.end_lineno is None:
        return source_text

    lines = source_text.splitlines(keepends=True)
    start = guard.lineno - 1
    end = guard.end_lineno
    replacement = ["def main() -> None:\n", *lines[start + 1:end], "\n\nif __name__ == \"__main__\":\n", "    main()\n"]
    return "".join([*lines[:start], *replacement, *lines[end:]])


def cython_source_for_target(target: Path, build_root: Path) -> Path:
    source_text = target.read_text(encoding="utf-8")
    rewritten = rewrite_main_guard_as_main_function(source_text)
    if rewritten == source_text:
        return target

    transformed = build_root / "sources" / relative(target)
    transformed.parent.mkdir(parents=True, exist_ok=True)
    transformed.write_text(rewritten, encoding="utf-8")
    return transformed


def strip_command_for_platform(extension: Path) -> list[str]:
    if sys.platform == "darwin":
        return ["strip", "-x", str(extension)]
    return ["strip", "--strip-unneeded", str(extension)]


def strip_native_extensions(
    output_dir: Path,
    extension_files: list[str],
    required: bool = False,
) -> list[str]:
    if not extension_files:
        return []
    if shutil.which("strip") is None:
        if required:
            raise SystemExit("strip not found; production protected artifacts require stripped native extensions.")
        print("Warning: strip not found; native extension symbols were not stripped.")
        return []

    stripped: list[str] = []
    for rel in extension_files:
        extension = output_dir / rel
        try:
            run_command(strip_command_for_platform(extension))
        except subprocess.CalledProcessError as exc:
            raise SystemExit(exc.stdout or str(exc)) from exc
        stripped.append(rel)
    return stripped


def compile_cython_targets(
    cython_targets: list[Path],
    output_dir: Path,
    pyproject: dict,
    check_only: bool = False,
) -> list[str]:
    if not cython_targets:
        raise SystemExit(
            "No Cython targets configured. Add [tool.protected-artifacts].cython-targets."
        )

    if check_only:
        for target in cython_targets:
            print(f"+ {sys.executable} <cython-build> {relative(target)} as {cython_module_name(target, pyproject)}")
        return []

    build_root = REPO_ROOT / "build" / "protected-cython"
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    setup_path = build_root / "setup_cython.py"
    build_temp = build_root / "temp"
    cython_build = build_root / "cythonized"
    build_lib = cython_build_lib_root(output_dir, cython_targets, pyproject)

    modules = []
    for target in cython_targets:
        modules.append(
            {
                "name": cython_module_name(target, pyproject),
                "source": str(cython_source_for_target(target, build_root)),
            }
        )
    setup_path.write_text(
        textwrap.dedent(
            f"""
            from setuptools import Extension, setup
            from Cython.Build import cythonize

            modules = {modules!r}
            setup(
                ext_modules=cythonize(
                    [Extension(item["name"], [item["source"]]) for item in modules],
                    compiler_directives={{
                        "language_level": "3",
                        "embedsignature": False,
                        "emit_code_comments": False,
                    }},
                    build_dir={str(cython_build)!r},
                )
            )
            """
        ).lstrip(),
        encoding="utf-8",
    )

    try:
        run_command([
            sys.executable,
            str(setup_path),
            "build_ext",
            "--build-lib",
            str(build_lib),
            "--build-temp",
            str(build_temp),
        ])
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stdout or str(exc)) from exc
    finally:
        for target in cython_targets:
            generated_c = target.with_suffix(".c")
            if generated_c.exists():
                generated_c.unlink()

    compiled = assert_cython_extensions_exist(output_dir, cython_targets)
    shutil.rmtree(build_root, ignore_errors=True)
    return compiled


def assert_output_is_protected(
    output_dir: Path,
    allowed_plaintext_python_files: list[str],
) -> list[str]:
    allowed_plaintext_python_set = set(allowed_plaintext_python_files)
    unexpected_plaintext: list[str] = []
    plaintext: list[str] = []
    for source in sorted(output_dir.rglob("*.py")):
        rel = source.relative_to(output_dir).as_posix()
        if rel in allowed_plaintext_python_set:
            plaintext.append(rel)
        else:
            unexpected_plaintext.append(rel)
    if unexpected_plaintext:
        sample = "\n  ".join(unexpected_plaintext[:20])
        raise SystemExit(
            "Protected output contains unexpected plaintext Python files:\n"
            f"  {sample}"
        )
    return plaintext


def assert_no_forbidden_artifacts(output_dir: Path) -> None:
    forbidden: list[str] = []
    for source in sorted(output_dir.rglob("*")):
        if source.is_dir():
            continue
        rel = source.relative_to(output_dir).as_posix()
        lowered = rel.lower()
        if source.suffix in FORBIDDEN_ARTIFACT_SUFFIXES:
            forbidden.append(rel)
            continue
        if "pyarmor" in lowered or "pytransform" in lowered:
            forbidden.append(rel)
            continue
        if source.suffix == ".py":
            content = source.read_text(encoding="utf-8", errors="ignore")
            if "__pyarmor__" in content or "pyarmor_runtime" in content or "pytransform" in content:
                forbidden.append(rel)
    if forbidden:
        sample = "\n  ".join(forbidden[:20])
        raise SystemExit(
            "Protected output contains forbidden build/protection artifacts:\n"
            f"  {sample}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_file_records(output_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for source in sorted(output_dir.rglob("*")):
        if source.is_dir() or source.name == MANIFEST_FILENAME:
            continue
        rel = source.relative_to(output_dir).as_posix()
        records.append(
            {
                "path": rel,
                "sha256": sha256_file(source),
                "size": source.stat().st_size,
            }
        )
    return records


def is_native_audit_file(path: Path) -> bool:
    name = path.name
    return (
        any(name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
        or path.suffix in NATIVE_AUDIT_SUFFIXES
    )


def native_strings(path: Path) -> list[str]:
    if shutil.which("strings") is not None:
        completed = subprocess.run(
            ["strings", "-a", str(path)],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            return completed.stdout.splitlines()

    data = path.read_bytes()
    values = re.findall(rb"[ -~]{8,}", data)
    return [value.decode("utf-8", errors="ignore") for value in values]


def redact_secret(value: str) -> str:
    if len(value) <= 12:
        return "<redacted>"
    return f"{value[:4]}...{value[-4:]}"


def redacted_context(context: str, value: str) -> str:
    compact = context.strip()
    if value:
        compact = compact.replace(value, redact_secret(value))
    if len(compact) > 180:
        compact = f"{compact[:177]}..."
    return compact


def is_audit_allowlisted(
    rel: str,
    rule: str,
    value: str,
    context: str,
    allowlist: list[re.Pattern[str]],
) -> bool:
    haystack = "\n".join([rel, rule, value, context])
    return any(pattern.search(haystack) for pattern in allowlist)


def looks_like_placeholder(value: str) -> bool:
    cleaned = value.strip().strip('"\'')
    lowered = cleaned.lower()
    if not cleaned:
        return True
    if lowered in {"false", "none", "null", "redacted", "true"}:
        return True
    if lowered.startswith(("env.", "getenv(", "os.getenv(", "self.", "settings.")):
        return True
    if cleaned.startswith(("$", "${", "<", "{")):
        return True
    if cleaned.upper() == cleaned and "_" in cleaned:
        return True
    return False


def looks_like_secret_value(value: str) -> bool:
    cleaned = value.strip().strip('"\'')
    if looks_like_placeholder(cleaned):
        return False
    if len(cleaned) < 16:
        return False
    classes = sum(
        1
        for chars in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[^A-Za-z0-9]")
        if re.search(chars, cleaned)
    )
    if len(cleaned) >= 24 and re.fullmatch(r"[A-Za-z0-9_./+=:-]+", cleaned) and classes >= 2:
        return True
    return classes >= 3


def sensitive_findings_for_text(
    rel: str,
    location: str,
    text: str,
    allowlist: list[re.Pattern[str]],
) -> list[str]:
    findings: list[str] = []

    def add(rule: str, value: str) -> None:
        if is_audit_allowlisted(rel, rule, value, text, allowlist):
            return
        findings.append(f"{rel}:{location} [{rule}] {redacted_context(text, value)}")

    for rule, pattern in KNOWN_SECRET_PATTERNS:
        for match in pattern.finditer(text):
            add(rule, match.group(0))

    for match in INTERNAL_URL_RE.finditer(text):
        add("internal url", match.group(0))

    for match in WEAK_PASSWORD_ASSIGNMENT_RE.finditer(text):
        add("weak default password", match.group(1))

    for match in WEAK_PASSWORD_FALLBACK_RE.finditer(text):
        add("weak default password", match.group(1))

    for match in SECRET_ASSIGNMENT_RE.finditer(text):
        value = match.group(1)
        if looks_like_secret_value(value):
            add("hardcoded secret", value)

    return findings


def audit_sensitive_artifacts(output_dir: Path, pyproject: dict) -> list[str]:
    if not output_dir.exists() or not output_dir.is_dir():
        raise SystemExit(f"Protected output directory does not exist: {output_dir}")

    allowlist = sensitive_audit_allowlist(pyproject)
    findings: list[str] = []
    for source in sorted(output_dir.rglob("*")):
        if source.is_dir():
            continue
        rel = source.relative_to(output_dir).as_posix()
        if source.suffix in TEXT_AUDIT_SUFFIXES:
            content = source.read_text(encoding="utf-8", errors="ignore")
            for line_no, line in enumerate(content.splitlines(), 1):
                findings.extend(sensitive_findings_for_text(rel, str(line_no), line, allowlist))
        elif is_native_audit_file(source):
            for value in native_strings(source):
                findings.extend(sensitive_findings_for_text(rel, "strings", value, allowlist))

    if findings:
        sample = "\n  ".join(findings[:20])
        raise SystemExit(
            "Protected output failed sensitive string audit:\n"
            f"  {sample}"
        )
    return findings


def remove_manifest(output_dir: Path) -> bool:
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return False
    manifest_path.unlink()
    return True


def verify_existing_output(
    output_dir: Path,
    cython_targets: list[Path],
    allowed_plaintext_python_files: list[str],
) -> tuple[list[str], list[str]]:
    if not output_dir.exists() or not output_dir.is_dir():
        raise SystemExit(f"Protected output directory does not exist: {output_dir}")
    cython_extension_files = assert_cython_extensions_exist(output_dir, cython_targets)
    plaintext_files = assert_output_is_protected(output_dir, allowed_plaintext_python_files)
    assert_no_forbidden_artifacts(output_dir)

    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_records = manifest.get("artifact_files")
        if expected_records is not None:
            current_records = artifact_file_records(output_dir)
            if expected_records != current_records:
                raise SystemExit("Protected artifact manifest does not match current output files")
    return cython_extension_files, plaintext_files


def write_manifest(
    output_dir: Path,
    pyproject: dict,
    targets: list[Path],
    cython_extension_files: list[str],
    plaintext_runtime_python_files: list[str],
    copied_runtime_files: list[str],
    stripped_extension_files: list[str],
    slimmed_plaintext_python_files: list[str],
    archive_path: Path | None,
) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": project_name(pyproject),
        "version": project_version(pyproject),
        "protection_tool": "cython",
        "python": sys.version.split()[0],
        "python_abi_tags": sorted(
            {
                path.split(".cpython-", 1)[1].rsplit(".", 1)[0]
                for path in cython_extension_files
                if ".cpython-" in path
            }
        ),
        "targets": [str(relative(path)) for path in targets],
        "cython_extension_files": cython_extension_files,
        "plaintext_runtime_python_files": plaintext_runtime_python_files,
        "copied_runtime_files": copied_runtime_files,
        "stripped_extension_files": stripped_extension_files,
        "slimmed_plaintext_python_files": slimmed_plaintext_python_files,
        "artifact_files": artifact_file_records(output_dir),
        "archive": str(archive_path.relative_to(REPO_ROOT)) if archive_path else None,
        "note": "Deploy this artifact instead of the source checkout.",
    }
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def archive_path(pyproject: dict) -> Path:
    archive_dir = REPO_ROOT / "dist-protected"
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir / f"{project_name(pyproject)}-{project_version(pyproject)}-protected.tar.gz"


def create_archive(output_dir: Path, archive: Path) -> Path:
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)
    return archive


def main() -> int:
    args = parse_args()
    pyproject = load_pyproject()
    output_dir = (REPO_ROOT / args.output).resolve()

    if args.audit_only:
        audit_sensitive_artifacts(output_dir, pyproject)
        print(f"Sensitive string audit passed: {output_dir.relative_to(REPO_ROOT)}")
        return 0

    targets = discover_targets(pyproject, args.target)
    cython_targets = configured_cython_targets(pyproject)
    plaintext_runtime_python_files = plaintext_runtime_py_files(pyproject)
    plaintext_runtime_python_rel = [
        relative(path).as_posix()
        for path in plaintext_runtime_python_files
    ]
    if not targets:
        raise SystemExit("No Python runtime targets found to protect")

    assert_no_unprotected_runtime_python(
        targets,
        args.allow_plaintext_py,
        cython_targets,
        plaintext_runtime_python_files,
    )

    print("Runtime targets:")
    for target in targets:
        print(f"  - {relative(target)}")
    print("Cython targets:")
    for target in cython_targets:
        print(f"  - {relative(target)}")
    if plaintext_runtime_python_files:
        print("Plaintext runtime Python files:")
        for target in plaintext_runtime_python_files:
            print(f"  - {relative(target)}")

    if args.check_only:
        compile_cython_targets(cython_targets, output_dir, pyproject, check_only=True)
        print(f"Check passed. Protected tree would be: {output_dir.relative_to(REPO_ROOT)}")
        return 0

    if args.verify_only:
        cython_extension_files, plaintext_files = verify_existing_output(
            output_dir,
            cython_targets,
            plaintext_runtime_python_rel,
        )
        audit_sensitive_artifacts(output_dir, pyproject)
        print(f"Verified protected tree: {output_dir.relative_to(REPO_ROOT)}")
        print(f"Sensitive string audit passed: {output_dir.relative_to(REPO_ROOT)}")
        print(f"Cython extension files: {len(cython_extension_files)}")
        if plaintext_files:
            print(f"Plaintext runtime Python files: {len(plaintext_files)}")
        return 0

    clean_output(output_dir)
    copied_runtime_files = copy_runtime_files(
        output_dir,
        targets,
        runtime_include_globs(pyproject),
        runtime_exclude_globs(pyproject),
        plaintext_runtime_python_files,
    )
    try:
        cython_extension_files = compile_cython_targets(cython_targets, output_dir, pyproject)
        stripped_extension_files = [] if args.no_strip else strip_native_extensions(
            output_dir,
            cython_extension_files,
            required=args.production,
        )
        plaintext_files = assert_output_is_protected(output_dir, plaintext_runtime_python_rel)
        slimmed_plaintext_files = slim_plaintext_runtime_python(output_dir, plaintext_files)
        assert_no_forbidden_artifacts(output_dir)
        audit_sensitive_artifacts(output_dir, pyproject)
    except BaseException:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise

    archive = None if args.no_archive else archive_path(pyproject)
    write_manifest(
        output_dir,
        pyproject,
        targets,
        cython_extension_files,
        plaintext_files,
        copied_runtime_files,
        stripped_extension_files,
        slimmed_plaintext_files,
        archive,
    )
    production_manifest_removed = False
    if args.production:
        verify_existing_output(
            output_dir,
            cython_targets,
            plaintext_runtime_python_rel,
        )
        production_manifest_removed = remove_manifest(output_dir)
    if archive:
        create_archive(output_dir, archive)

    print(f"Protected tree: {output_dir.relative_to(REPO_ROOT)}")
    if archive:
        print(f"Protected archive: {archive.relative_to(REPO_ROOT)}")
    print(f"Cython extension files: {len(cython_extension_files)}")
    if plaintext_files:
        print(f"Plaintext runtime Python files: {len(plaintext_files)}")
    if slimmed_plaintext_files:
        print(f"Slimmed plaintext runtime Python files: {len(slimmed_plaintext_files)}")
    print(f"Copied runtime files: {len(copied_runtime_files)}")
    if production_manifest_removed:
        print(f"Production manifest removed: {MANIFEST_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
