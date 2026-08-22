"""Fail the build if a credential-shaped string is about to be committed.

This exists because it nearly happened: `.env.example` is a tracked template, and filling
it in with real values instead of copying it to `.env` first is a natural mistake that
publishes a Databricks token and a Supabase service key to a public repository.

Runs in CI on every push and pull request, and works as a pre-commit hook:

    python scripts/check_no_secrets.py            # scan tracked files
    python scripts/check_no_secrets.py --staged   # scan what is staged (hook mode)

Detection is by shape, not by wordlist, so it catches a pasted key regardless of the
variable it was pasted into.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Credential formats this project actually handles. Each is anchored on a distinctive
# prefix or structure so ordinary code and prose do not trip it.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Databricks personal access token", re.compile(r"\bdapi[0-9a-f]{32}\b")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Postgres URI with inline password", re.compile(r"postgres(?:ql)?://[^\s:@/]+:[^\s@/]+@")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]

# Supabase keys are JWTs, and which one it is matters entirely. The `anon` key is
# designed to be public -- it ships in client-side JavaScript on every app built on the
# platform, and under row-level security it can only read. The `service_role` key bypasses
# RLS completely. Both look identical to a regex, so decode the payload and read the role
# claim rather than banning the shape.
JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.([A-Za-z0-9_-]{10,})\.[A-Za-z0-9_-]*")
PUBLISHABLE_ROLES = {"anon"}


def jwt_role(payload_segment: str) -> str | None:
    """Role claim from a JWT payload, or None if it will not decode."""
    import base64
    import json

    padded = payload_segment + "=" * (-len(payload_segment) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(padded)).get("role")
    except Exception:
        return None

# Placeholder markers. A template value must look obviously fake.
PLACEHOLDER = re.compile(
    r"(x{4,}|your[-_]|<[^>]+>|example\.com|changeme|\.\.\.)", re.IGNORECASE
)

# Values that are configuration rather than credentials and may appear verbatim.
BENIGN_VALUES = {"5432", "require", "prefer", "disable", "themepark", "7", "30", "14"}

SKIP_DIRS = {".git", "artifacts", "site", "trained_models", "mlruns", "__pycache__", "legacy"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".parquet", ".pyc", ".zip", ".mp4"}

# This file necessarily contains the patterns it searches for.
SELF = "scripts/check_no_secrets.py"


def tracked_files(staged: bool) -> list[str]:
    cmd = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
        if staged
        else ["git", "ls-files"]
    )
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f.strip()]


def should_scan(path: str) -> bool:
    if path == SELF:
        return False
    parts = Path(path).parts
    if any(p in SKIP_DIRS for p in parts):
        return False
    return Path(path).suffix.lower() not in SKIP_SUFFIXES


def scan_content(path: str, text: str) -> list[str]:
    findings = []
    for label, pattern in PATTERNS:
        for match in pattern.finditer(text):
            line = text[: match.start()].count("\n") + 1
            findings.append(f"{path}:{line}  {label}")

    for match in JWT.finditer(text):
        role = jwt_role(match.group(1))
        if role in PUBLISHABLE_ROLES:
            continue  # the anon key is meant to be published
        line = text[: match.start()].count("\n") + 1
        described = f"role={role}" if role else "role could not be decoded"
        findings.append(f"{path}:{line}  JWT that is not the public anon key ({described})")

    return findings


def scan_env_template(path: str, text: str) -> list[str]:
    """A tracked `.env*` template must contain only obvious placeholders.

    Catches the specific mistake this script was written for: real values typed into the
    committed template instead of into the gitignored `.env`.
    """
    findings = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("\"'")
        if not value or value in BENIGN_VALUES or PLACEHOLDER.search(value):
            continue
        findings.append(
            f"{path}:{number}  {key.strip()} holds a real-looking value "
            f"({len(value)} chars) — templates must contain placeholders only"
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true", help="scan staged changes only")
    args = parser.parse_args()

    findings: list[str] = []
    for path in tracked_files(args.staged):
        if not should_scan(path):
            continue
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue

        findings.extend(scan_content(path, text))
        if Path(path).name.startswith(".env"):
            findings.extend(scan_env_template(path, text))

    if findings:
        print("SECRET SCAN FAILED\n", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        print(
            "\nMove real values into .env (gitignored) and keep the tracked template "
            "as placeholders.\nIf a secret was already pushed, rotate it — removing the "
            "file does not un-publish it.",
            file=sys.stderr,
        )
        return 1

    print("secret scan clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
