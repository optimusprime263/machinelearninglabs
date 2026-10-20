"""Print Python and library versions; save them to lab03/results/versions.txt.

Run from the repo root:  python lab03/src/print_versions.py
"""
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent.parent / "results" / "versions.txt"


def main():
    lines = [f"Python {sys.version.split()[0]}"]
    for req in (ROOT / "requirements.txt").read_text().splitlines():
        req = req.strip()
        if not req or req.startswith("#"):
            continue
        name, _, pinned = req.partition("==")
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "not installed"
        lines.append(f"{name:<16} installed {installed:<26} pinned {pinned}")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)


if __name__ == "__main__":
    main()
