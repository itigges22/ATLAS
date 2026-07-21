"""Per-line structural-importance signal.

Faithful port of wavescope-mcp `src/signal.ts`. Produces a per-line score from
indentation, structural keywords, and decorators, with comment / string-literal
awareness. Scores are clamped to [0, 2].
"""

from __future__ import annotations

import math
import re
from typing import List

from .language import LanguageConfig


# Token splitter: splits on whitespace, brackets, braces, parens, commas,
# semicolons, colons, quotes, backticks, arithmetic, logical, and bitwise
# operators. Does NOT split on `.`, `-`, `?`, `!` — so `obj.class` stays one
# token (preventing member-access keyword leaks) and `extend-type`,
# `defmulti`, `defined?` survive as single tokens.
_TOKEN_SPLIT = re.compile(r"[\s()\[\]{},;:'\"`=<>+*/&|^~%@#\\]+")


def _mask_string_literals(line: str, lang: LanguageConfig) -> str:
    """Mask the interior of single-line string/char literals with spaces so
    downstream comment / token detection ignores their contents. Handles
    single-quote, double-quote, and backtick literals with backslash escapes."""
    # In Rust `'a` is a lifetime and in Clojure `'` is the quote reader macro —
    # neither delimits a string, so treating `'` as one would mask real code.
    single_quote_is_string = lang.name not in ("rust", "clojure")
    # Backtick delimits strings/templates in JS/TS/Go/shell, but in Python
    # and Clojure it is not a string delimiter — a stray backtick there would
    # mask the rest of the line and feed the terminator-swallowing class of
    # bug.
    backtick_is_string = lang.name not in ("python", "clojure")
    chars = list(line)
    i = 0
    n = len(chars)
    while i < n:
        c = chars[i]
        if c == '"' or (backtick_is_string and c == "`") or (single_quote_is_string and c == "'"):
            quote = c
            j = i + 1
            while j < n:
                if chars[j] == "\\" and j + 1 < n:
                    chars[j] = " "
                    chars[j + 1] = " "
                    j += 2
                    continue
                if chars[j] == quote:
                    break
                chars[j] = " "
                j += 1
            i = j + 1
            continue
        i += 1
    return "".join(chars)


def _tokenize(line: str) -> List[str]:
    return _TOKEN_SPLIT.split(line)


def _score_line(raw: str, stripped: str, raw_indent: int, lang: LanguageConfig) -> float:
    """Score a single line of actual code (no comments)."""
    score = 0.0

    # Strip string literals first so quoted `//` URLs and quoted `/*` don't
    # confuse keyword / inline-comment scanning.
    code_only = _mask_string_literals(stripped, lang)

    # Strip inline single-line comment suffix.
    for prefix in lang.comment_prefixes:
        idx = -1
        frm = 0
        while frm < len(code_only):
            nxt = code_only.find(prefix, frm)
            if nxt == -1:
                break
            # PHP 8 attribute: `#[...]` is not a comment.
            if prefix == "#" and nxt + 1 < len(code_only) and code_only[nxt + 1] == "[":
                frm = nxt + 1
                continue
            idx = nxt
            break
        if idx != -1:
            code_only = code_only[:idx].strip()
            break

    # Indent: expand tabs as 4 spaces so tab-indented files score comparably
    # to space-indented ones.
    leading = raw[:raw_indent]
    expanded_indent = 0
    for ch in leading:
        expanded_indent += 4 if ch == "\t" else 1
    indent_level = min(expanded_indent / 4, 8)
    score += indent_level * lang.indent_weight

    for token in _tokenize(code_only):
        if not token:
            continue
        if token in lang.structural_keywords:
            score += lang.structural_keywords[token]

    # Decorators / annotations: line-start `@`, Rust `#[...]`, PHP 8 `#[...]`,
    # or inline `@Annotation` (e.g. Java `public @Nullable String foo()`).
    if lang.decorator_weight > 0:
        starts_with_at = code_only.startswith("@")
        rust_attr = lang.name == "rust" and code_only.startswith("#[")
        php_attr = lang.name == "php" and code_only.startswith("#[")
        inline_at = re.search(r"(^|\s)@[A-Za-z_]", code_only) is not None
        if starts_with_at or rust_attr or php_attr or inline_at:
            score += lang.decorator_weight

    return min(score, 2.0)


def compute_signal(lines: List[str], lang: LanguageConfig) -> List[float]:
    """Compute the per-line structural signal for a complete file.

    NOTE: callers must pass a complete file (all lines), not a slice — the
    block-comment / docstring state machine spans lines but is initialized
    fresh per call, so a partial file would misclassify comments.
    """
    signal: List[float] = [0.0] * len(lines)
    in_block_comment = False
    block_comment_depth = 0
    in_doc_string = False
    doc_string_delim = None

    for i, rawline in enumerate(lines):
        raw = rawline
        trimmed = raw.lstrip()
        indent = len(raw) - len(trimmed)
        stripped = trimmed.rstrip()
        # String-masked version used for comment/keyword detection; original
        # line drives indent/length calculations.
        masked = _mask_string_literals(stripped, lang)

        # ── Continuation of multiline comments / docstrings ──
        # Terminator scanning runs on the UNMASKED text: comment interiors
        # are prose, and masking them treats an apostrophe in "Don't" as a
        # string opener that swallows the `*/` (or `\"\"\"`) terminator —
        # permanently zeroing the rest of the file's signal.
        if in_block_comment:
            if lang.block_comment_uses_paren_depth:
                depth = block_comment_depth
                close_idx = -1
                for ci, mc in enumerate(stripped):
                    if mc == "(":
                        depth += 1
                    elif mc == ")":
                        depth -= 1
                        if depth == 0:
                            close_idx = ci
                            break
                if close_idx != -1:
                    in_block_comment = False
                    block_comment_depth = 0
                    after = stripped[close_idx + 1:]
                    signal[i] = _score_line(raw, after.strip(), indent, lang) if after.strip() else 0.0
                else:
                    block_comment_depth = depth
                    signal[i] = 0.0
            else:
                end_idx = stripped.find(lang.block_comment_end)
                if end_idx != -1:
                    in_block_comment = False
                    after = stripped[end_idx + len(lang.block_comment_end):]
                    signal[i] = _score_line(raw, after.strip(), indent, lang) if after.strip() else 0.0
                else:
                    signal[i] = 0.0
            continue

        if in_doc_string:
            end_idx = stripped.find(doc_string_delim)
            if end_idx != -1:
                in_doc_string = False
                doc_string_delim = None
                after = stripped[end_idx + 3:]
                signal[i] = _score_line(raw, after.strip(), indent, lang) if after.strip() else 0.0
            else:
                signal[i] = 0.0
            continue

        # ── Detect new Python docstrings (triple quotes) ──
        # Scanned on the unmasked stripped line because _mask_string_literals
        # would have consumed them as regular strings.
        if lang.name == "python":
            # Cut at a line comment first (found on the masked text so a `#`
            # inside a real string doesn't cut): a comment like
            # `# see """ usage` must not flip the docstring state machine.
            hash_pos = masked.find("#")
            doc_scan = stripped if hash_pos == -1 else stripped[:hash_pos]
            dq = doc_scan.find('"""')
            sq = doc_scan.find("'''")
            dq_idx = min(
                dq if dq != -1 else math.inf,
                sq if sq != -1 else math.inf,
            )
            if math.isfinite(dq_idx):
                dq_idx = int(dq_idx)
                delim = doc_scan[dq_idx:dq_idx + 3]
                before = doc_scan[:dq_idx].strip()
                after = doc_scan[dq_idx + 3:]
                has_closing = delim in after

                if has_closing:
                    real = "; ".join(
                        p for p in [before, after.replace(delim, "").strip()] if p
                    )
                    signal[i] = _score_line(raw, real, indent, lang) if real else 0.0
                    continue
                in_doc_string = True
                doc_string_delim = delim
                signal[i] = _score_line(raw, before, indent, lang) if before else 0.0
                continue

        # ── Detect block comment start (anywhere on line, including inline) ──
        if lang.block_comment_at_line_start:
            bc_start_idx = 0 if masked.startswith(lang.block_comment_start) else -1
        else:
            bc_scan = masked
            if lang.name == "python":
                # Same comment cut as the docstring branch: a `"""` after a
                # line comment (`# see """ usage`) must not open python's
                # redundant block-comment path either.
                hp = masked.find("#")
                if hp != -1:
                    bc_scan = masked[:hp]
            bc_start_idx = bc_scan.find(lang.block_comment_start)
        if bc_start_idx != -1:
            before = stripped[:bc_start_idx].strip()
            # Close-scan on the RAW interior: the comment's contents are
            # prose, and masking them lets an apostrophe swallow the
            # terminator (`/* Don't use */` would never close).
            after_delim = stripped[bc_start_idx + len(lang.block_comment_start):]
            after_delim_raw = after_delim

            end_idx = -1
            if lang.block_comment_uses_paren_depth:
                depth = 1
                for ci, mc in enumerate(after_delim):
                    if mc == "(":
                        depth += 1
                    elif mc == ")":
                        depth -= 1
                        if depth == 0:
                            end_idx = ci
                            break
            else:
                end_idx = after_delim.find(lang.block_comment_end)

            if end_idx != -1:
                after = after_delim_raw[end_idx + len(lang.block_comment_end):].strip()
                real = "; ".join(p for p in [before, after] if p)
                signal[i] = _score_line(raw, real, indent, lang) if real else 0.0
                continue
            in_block_comment = True
            block_comment_depth = 1 if lang.block_comment_uses_paren_depth else 0
            signal[i] = _score_line(raw, before, indent, lang) if before else 0.0
            continue

        # ── Single-line comments ──
        def _is_comment(p: str) -> bool:
            if not masked.startswith(p):
                return False
            # PHP 8 attributes: `#[...]` is NOT a comment.
            if p == "#" and masked.startswith("#["):
                return False
            return True

        if any(_is_comment(p) for p in lang.comment_prefixes) or len(masked) == 0:
            signal[i] = 0.0
            continue

        signal[i] = _score_line(raw, stripped, indent, lang)

    return signal
