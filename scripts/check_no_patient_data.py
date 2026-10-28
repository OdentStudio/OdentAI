#!/usr/bin/env python3
"""Refuse anything that looks like patient or doctor data. Run by the git hooks.

No doctor and no patient data in this repository, ever: not a scan, not a
path, not a log line, not a name. Clinic folders are named after patients
(surname, first name, dates, planned teeth, post-op), so even a file path
is personal health data. The repository is public: what gets pushed can be
copied by anyone and cannot be taken back, so it is stopped before the commit.

This file holds no names on purpose — it looks for the shapes such data
takes. A hit means: remove it, do not bypass the hook.

    python scripts/check_no_patient_data.py --staged       # pre-commit
    python scripts/check_no_patient_data.py --range A..B   # pre-push (or: B --not --remotes)
    python scripts/check_no_patient_data.py --all          # every tracked file
"""
import re
import subprocess
import sys

PATTERNS = [
    (r"(?i)[A-Z]:[\\/]+Dental[\\/]", "a clinic data folder (C:\\Dental\\...)"),
    (r"(?i)/media/[^/\s]+/[0-9A-F]{8,}/Dental", "a clinic data drive"),
    (r"(?i)(^|[\\/\s\"'`])(Guides|Kacper|Dmitry)[\\/]", "a doctor's case folder"),
    (r"(?i)[\\/](post_?op|postop)[\\/_ ]", "a post-op case folder"),
    (r"\b20\d{6}_\d{6}_\d{3}\b", "a DICOM export folder (date_time_ms)"),
    (r"(^|[\\/\s\"'`])\d{2}[ _]\d{2}[ _]2\d[ _][^\s\\/]*[A-Za-zА-Яа-яЁё]{3,}", "a dated case folder (dd mm yy name)"),
    (r"[А-ЯЁ][а-яё]+ [А-ЯЁ]{1,2}\.(dcm|zip|mha|nii)", "a scan file named after a person"),
    (r"(?i)\b(соге|одое)\b", "a case-folder tag"),
]
# The add-on's own authors are named in bl_info on purpose; that is attribution,
# not data. Lines matching these are skipped.
ALLOW = [r"Dr\. [A-Z]", r"AUTHORS = \["]
SELF = "scripts/check_no_patient_data.py"


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout


def added_lines(diff: str):
    """(file, line) for every added line of a unified diff."""
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+") and not line.startswith("+++") and path != SELF:
            yield path, line[1:]
        elif line.startswith("diff --git"):
            path = None


def scan(pairs):
    hits = []
    for path, text in pairs:
        # Paths themselves leak too: check the file name as well as the line.
        for what in (path or "", text):
            if any(re.search(a, what) for a in ALLOW):
                continue
            for pat, why in PATTERNS:
                if re.search(pat, what):
                    hits.append((path, why, what.strip()[:120]))
                    break
    return hits


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--staged"
    if mode == "--staged":
        pairs = added_lines(git("diff", "--cached", "-U0", "--no-color"))
        pairs = list(pairs) + [(p, "") for p in git("diff", "--cached", "--name-only").splitlines()]
    elif mode == "--range":
        rev = sys.argv[2:]                  # "A..B", or "B --not --remotes" for a new branch
        pairs = list(added_lines(git("log", "-p", "--no-color", "--format=", *rev)))
        pairs += [(None, m) for m in git("log", "--format=%B", *rev).splitlines()]
    elif mode == "--all":
        pairs = []
        for p in git("ls-files").splitlines():
            if p == SELF:
                continue
            pairs.append((p, ""))
            try:
                with open(p, encoding="utf-8", errors="strict") as f:
                    pairs += [(p, l) for l in f]
            except (UnicodeDecodeError, OSError):
                pass                                        # binary
    else:
        print(__doc__)
        return 2
    hits = scan(pairs)
    if not hits:
        return 0
    print("REFUSED: this looks like patient or doctor data. None of it may enter "
          "this repository — not a path, not a log, not a name.", file=sys.stderr)
    for path, why, text in hits[:30]:
        print(f"  {path or '(commit message)'}: {why}: {text}", file=sys.stderr)
    print("Remove it (keep experiment logs outside the repo, name scans 'A', 'B'...). "
          "Do not bypass this check with --no-verify.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
