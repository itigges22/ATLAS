"""Per-language structural-keyword tables.

Faithful port of wavescope-mcp `src/language.ts`. Keyword weights, comment
delimiters, indent/decorator weights, and the detection order are preserved
verbatim so signal scoring matches upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class LanguageConfig:
    name: str
    extensions: List[str]
    structural_keywords: Dict[str, float]
    comment_prefixes: List[str]
    block_comment_start: str
    block_comment_end: str
    indent_weight: float
    decorator_weight: float
    # Filenames (no extension) that should also map here, e.g. "Rakefile".
    filenames: List[str] = field(default_factory=list)
    # If true, block comment delimiters must appear at the start of the line.
    block_comment_at_line_start: bool = False
    # If true, block comment end tracking uses paren-depth counting
    # (for Clojure-style (comment ...) forms with nested S-expressions).
    block_comment_uses_paren_depth: bool = False


# ─── Base keyword sets ───────────────────────────────────────

_C_LIKE: Dict[str, float] = {
    "class": 1.0,
    "export": 0.6,
    "import": 0.6,
    "public": 0.3,
    "private": 0.3,
    "protected": 0.3,
    "abstract": 0.4,
    "static": 0.3,
    "async": 0.3,
    "const": 0.3,
    "let": 0.2,
    "var": 0.2,
    "if": 0.3,
    "else": 0.2,
    "for": 0.3,
    "while": 0.3,
    "do": 0.2,
    "switch": 0.3,
    "case": 0.2,
    "default": 0.2,
    "try": 0.3,
    "catch": 0.3,
    "finally": 0.2,
    "return": 0.2,
    "throw": 0.2,
}

_JS_LIKE: Dict[str, float] = {**_C_LIKE, "function": 0.9}

_TS_LIKE: Dict[str, float] = {**_JS_LIKE, "interface": 0.9, "type": 0.5, "enum": 0.8}


# ─── Language configurations ─────────────────────────────────

python_config = LanguageConfig(
    name="python",
    extensions=[".py", ".pyi", ".pyx"],
    structural_keywords={
        "class": 1.0,
        "def": 0.9,
        "import": 0.6,
        "from": 0.5,
        "return": 0.2,
        "yield": 0.2,
        "raise": 0.2,
        "if": 0.3,
        "elif": 0.2,
        "else": 0.2,
        "try": 0.3,
        "except": 0.3,
        "finally": 0.2,
        "for": 0.3,
        "while": 0.3,
        "with": 0.4,
        "match": 0.3,
        "case": 0.2,
    },
    comment_prefixes=["#"],
    # Python docstrings (triple quotes) are handled specially in signal.py.
    block_comment_start='"""',
    block_comment_end='"""',
    indent_weight=0.15,
    decorator_weight=0.5,
)

ts_config = LanguageConfig(
    name="typescript",
    extensions=[".ts", ".tsx", ".mts", ".cts"],
    structural_keywords={**_TS_LIKE, "get": 0.3, "set": 0.3},
    comment_prefixes=["//"],
    block_comment_start="/*",
    block_comment_end="*/",
    indent_weight=0.15,
    decorator_weight=0.5,
)

js_config = LanguageConfig(
    name="javascript",
    extensions=[".js", ".jsx", ".mjs", ".cjs"],
    structural_keywords={**_JS_LIKE, "get": 0.3, "set": 0.3},
    comment_prefixes=["//"],
    block_comment_start="/*",
    block_comment_end="*/",
    indent_weight=0.15,
    decorator_weight=0.5,
)

go_config = LanguageConfig(
    name="go",
    extensions=[".go"],
    structural_keywords={
        **_C_LIKE,
        "func": 0.9,
        "go": 0.2,
        "defer": 0.2,
        "select": 0.2,
        "struct": 0.9,
        "package": 0.3,
    },
    comment_prefixes=["//"],
    block_comment_start="/*",
    block_comment_end="*/",
    indent_weight=0.1,
    decorator_weight=0.0,
)

rust_config = LanguageConfig(
    name="rust",
    extensions=[".rs"],
    structural_keywords={
        **_C_LIKE,
        "fn": 0.9,
        "impl": 0.9,
        "mod": 0.6,
        "use": 0.5,
        "pub": 0.4,
        "mut": 0.1,
        "trait": 0.9,
        "struct": 0.9,
        "enum": 0.8,
        "type": 0.5,
        "match": 0.3,
        "where": 0.2,
        "unsafe": 0.3,
        "extern": 0.3,
        "macro_rules": 0.7,
    },
    comment_prefixes=["//"],
    block_comment_start="/*",
    block_comment_end="*/",
    indent_weight=0.15,
    decorator_weight=0.4,
)

java_config = LanguageConfig(
    name="java",
    extensions=[".java"],
    structural_keywords={
        **_TS_LIKE,
        "package": 0.6,
        "extends": 0.5,
        "implements": 0.5,
        "throws": 0.2,
        "synchronized": 0.2,
        "volatile": 0.1,
        "transient": 0.1,
        "native": 0.1,
        "strictfp": 0.1,
    },
    comment_prefixes=["//"],
    block_comment_start="/*",
    block_comment_end="*/",
    indent_weight=0.15,
    decorator_weight=0.5,
)

ruby_config = LanguageConfig(
    name="ruby",
    extensions=[".rb", ".rake", ".gemspec"],
    filenames=["Rakefile", "Gemfile"],
    structural_keywords={
        "class": 1.0,
        "def": 0.9,
        "module": 0.9,
        "require": 0.6,
        "include": 0.4,
        "extend": 0.4,
        "private": 0.3,
        "protected": 0.3,
        "public": 0.3,
        "attr_accessor": 0.5,
        "attr_reader": 0.5,
        "attr_writer": 0.5,
        "if": 0.3,
        "unless": 0.3,
        "else": 0.2,
        "elsif": 0.2,
        "while": 0.3,
        "until": 0.3,
        "for": 0.3,
        "do": 0.2,
        "begin": 0.3,
        "rescue": 0.3,
        "ensure": 0.2,
        "case": 0.2,
        "when": 0.2,
        "return": 0.2,
        "yield": 0.2,
        "raise": 0.2,
    },
    comment_prefixes=["#"],
    block_comment_start="=begin",
    block_comment_end="=end",
    indent_weight=0.12,
    decorator_weight=0.0,
    block_comment_at_line_start=True,
)

php_config = LanguageConfig(
    name="php",
    extensions=[".php"],
    structural_keywords={
        **_TS_LIKE,
        "namespace": 0.6,
        "use": 0.5,
        "trait": 0.8,
        "extends": 0.5,
        "implements": 0.5,
        "require_once": 0.5,
        "require": 0.5,
        "include": 0.4,
        "include_once": 0.4,
        "echo": 0.1,
    },
    comment_prefixes=["//", "#"],
    block_comment_start="/*",
    block_comment_end="*/",
    indent_weight=0.15,
    decorator_weight=0.4,
)

swift_config = LanguageConfig(
    name="swift",
    extensions=[".swift"],
    structural_keywords={
        **_C_LIKE,
        "func": 0.9,
        "guard": 0.3,
        "defer": 0.2,
        "protocol": 0.9,
        "extension": 0.7,
        "struct": 0.9,
        "actor": 0.9,
        "mutating": 0.3,
        "nonmutating": 0.3,
        "override": 0.3,
        "convenience": 0.2,
        "required": 0.2,
        "weak": 0.1,
        "unowned": 0.1,
        "throws": 0.2,
        "rethrows": 0.2,
        "associatedtype": 0.5,
        "typealias": 0.4,
    },
    comment_prefixes=["//"],
    block_comment_start="/*",
    block_comment_end="*/",
    indent_weight=0.15,
    decorator_weight=0.4,
)

kotlin_config = LanguageConfig(
    name="kotlin",
    extensions=[".kt", ".kts"],
    structural_keywords={
        **_C_LIKE,
        "fun": 0.9,
        "val": 0.2,
        "object": 0.7,
        "companion": 0.4,
        "data": 0.4,
        "sealed": 0.5,
        "open": 0.4,
        "override": 0.3,
        "suspend": 0.3,
        "operator": 0.3,
        "infix": 0.2,
        "inline": 0.2,
        "tailrec": 0.2,
        "external": 0.2,
        "annotation": 0.4,
        "expect": 0.3,
        "actual": 0.3,
    },
    comment_prefixes=["//"],
    block_comment_start="/*",
    block_comment_end="*/",
    indent_weight=0.15,
    decorator_weight=0.5,
)

scala_config = LanguageConfig(
    name="scala",
    extensions=[".scala", ".sc"],
    structural_keywords={
        **_C_LIKE,
        "def": 0.9,
        "val": 0.2,
        "object": 0.7,
        "trait": 0.9,
        "sealed": 0.5,
        "implicit": 0.4,
        "given": 0.3,
        "using": 0.3,
        "extension": 0.5,
        "opaque": 0.3,
        "case": 0.4,
        "match": 0.3,
        "lazy": 0.1,
        "override": 0.3,
    },
    comment_prefixes=["//"],
    block_comment_start="/*",
    block_comment_end="*/",
    indent_weight=0.15,
    decorator_weight=0.5,
)

clojure_config = LanguageConfig(
    name="clojure",
    extensions=[".clj", ".cljs", ".cljc", ".edn"],
    structural_keywords={
        "defn": 0.9,
        "defn-": 0.9,
        "def": 0.7,
        "defmacro": 0.9,
        "defmulti": 0.9,
        "defmethod": 0.8,
        "defprotocol": 0.9,
        "defrecord": 0.9,
        "deftype": 0.9,
        "definterface": 0.9,
        "defonce": 0.7,
        "extend-type": 0.8,
        "extend-protocol": 0.8,
        "letfn": 0.6,
        "reify": 0.6,
        "ns": 0.6,
        "require": 0.6,
        "use": 0.5,
        "import": 0.5,
        "fn": 0.4,
        "let": 0.2,
        "if": 0.3,
        "when": 0.3,
        "loop": 0.3,
        "for": 0.3,
        "doseq": 0.3,
        "try": 0.3,
        "catch": 0.3,
        "finally": 0.2,
    },
    comment_prefixes=[";"],
    block_comment_start="(comment",
    block_comment_end=")",
    indent_weight=0.12,
    decorator_weight=0.0,
    block_comment_uses_paren_depth=True,
)

generic_config = LanguageConfig(
    name="generic",
    extensions=[],
    structural_keywords={},
    comment_prefixes=["#"],
    block_comment_start="/*",
    block_comment_end="*/",
    indent_weight=0.1,
    decorator_weight=0.3,
)

# Ordered by priority — generic_config (last) is the fallback. It must remain
# last and must not share extensions with preceding configs.
configs: List[LanguageConfig] = [
    python_config,
    ts_config,
    js_config,
    go_config,
    rust_config,
    java_config,
    ruby_config,
    php_config,
    swift_config,
    kotlin_config,
    scala_config,
    clojure_config,
    generic_config,
]


def detect_language(filename: str) -> LanguageConfig:
    slash = max(filename.rfind("/"), filename.rfind("\\"))
    base = filename[slash + 1:] if slash >= 0 else filename
    for cfg in configs:
        if base in cfg.filenames:
            return cfg
    dot = base.rfind(".")
    ext = base[dot:].lower() if dot >= 0 else ""
    if ext:
        for cfg in configs:
            if ext in cfg.extensions:
                return cfg
    return generic_config
