#!/usr/bin/env python3
r"""Translate GitHub-style Markdown math to Doxygen math commands.

Doxygen's Markdown parser does not treat ```math fences as equations.  This
filter keeps the repository Markdown readable on GitHub while converting math
syntax to the \f$...\f$ and \f[...\f] forms that Doxygen/MathJax understands.
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_escaped(text: str, pos: int) -> bool:
    count = 0
    i = pos - 1
    while i >= 0 and text[i] == "\\":
        count += 1
        i -= 1
    return count % 2 == 1


def convert_inline_math_segment(text: str) -> str:
    dollars: list[int] = []
    for i, ch in enumerate(text):
        if ch != "$" or is_escaped(text, i):
            continue
        prev_is_dollar = i > 0 and text[i - 1] == "$"
        next_is_dollar = i + 1 < len(text) and text[i + 1] == "$"
        if prev_is_dollar or next_is_dollar:
            continue
        dollars.append(i)

    if len(dollars) < 2 or len(dollars) % 2 != 0:
        return text

    dollar_set = set(dollars)
    out: list[str] = []
    for i, ch in enumerate(text):
        if i in dollar_set:
            out.append(r"\f$")
        else:
            out.append(ch)
    return "".join(out)


def convert_inline_math(line: str) -> str:
    # Do not rewrite math delimiters inside inline code spans.
    parts = line.split("`")
    for i in range(0, len(parts), 2):
        parts[i] = convert_inline_math_segment(parts[i])
    return "`".join(parts)


def rewrite_doc_image_paths(line: str) -> str:
    return (
        line.replace('src="../../docs/figs/', 'src="')
        .replace('src="../docs/figs/', 'src="')
        .replace('src="docs/figs/', 'src="')
        .replace("](../../docs/figs/", "](")
        .replace("](../docs/figs/", "](")
        .replace("](docs/figs/", "](")
    )


def rewrite_doc_page_links(line: str) -> str:
    replacements = {
        "](examples/poisson-opt)": "](md_examples_2poisson-opt_2README.html)",
        "](examples/poisson)": "](md_examples_2poisson_2README.html)",
        "](apps/ns-forward)": "](md_apps_2ns-forward_2README.html)",
        'href="examples/poisson-opt"': 'href="md_examples_2poisson-opt_2README.html"',
        'href="examples/poisson"': 'href="md_examples_2poisson_2README.html"',
        'href="apps/ns-forward"': 'href="md_apps_2ns-forward_2README.html"',
    }
    for src, dst in replacements.items():
        line = line.replace(src, dst)
    return line


def is_root_readme(path: Path) -> bool:
    try:
        return path.resolve() == (Path.cwd() / "README.md").resolve()
    except OSError:
        return path.as_posix() == "README.md"


def convert(path: Path) -> str:
    out: list[str] = []
    in_code_fence = False
    in_math_fence = False
    in_dollar_display = False
    root_readme = is_root_readme(path)
    rewrote_root_title = False

    for raw in path.read_text(encoding="utf-8").splitlines(keepends=True):
        line = raw[:-1] if raw.endswith("\n") else raw
        newline = "\n" if raw.endswith("\n") else ""
        stripped = line.strip()

        if root_readme and not rewrote_root_title and stripped.startswith("# "):
            out.append("# Build and usage" + newline)
            rewrote_root_title = True
            continue

        if in_math_fence:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                out.append(r"\f]" + newline)
                in_math_fence = False
            else:
                out.append(raw)
            continue

        if in_dollar_display:
            if stripped == "$$":
                out.append(r"\f]" + newline)
                in_dollar_display = False
            else:
                out.append(raw)
            continue

        if stripped.startswith("```math") or stripped.startswith("~~~math"):
            out.append(r"\f[" + newline)
            in_math_fence = True
            continue

        if stripped == "$$" and not in_code_fence:
            out.append(r"\f[" + newline)
            in_dollar_display = True
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_fence = not in_code_fence
            out.append(raw)
            continue

        if in_code_fence:
            out.append(raw)
        else:
            line = convert_inline_math(line)
            line = rewrite_doc_image_paths(line)
            line = rewrite_doc_page_links(line)
            out.append(line + newline)

    return "".join(out)


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    sys.stdout.write(convert(Path(sys.argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
