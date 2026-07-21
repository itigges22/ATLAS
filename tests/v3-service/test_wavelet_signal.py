"""Conformance suite for the structural-signal port.

Translated from wavescope-mcp `src/signal.test.ts`.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

from wavelet.signal import compute_signal  # noqa: E402
from wavelet.language import detect_language  # noqa: E402

PY = detect_language("test.py")
TS = detect_language("test.ts")
GO = detect_language("test.go")


class TestPython:
    def test_zero_for_blank_and_comments(self):
        lines = ["", "  ", "# this is a comment", "   # indented comment"]
        assert compute_signal(lines, PY) == [0, 0, 0, 0]

    def test_zero_for_docstrings(self):
        lines = ['"""Module docstring"""', '"""', "Multi-line", "docstring", '"""', "def foo(): pass"]
        sig = compute_signal(lines, PY)
        assert sig[0] == 0
        assert sig[1] == 0
        assert sig[2] == 0
        assert sig[3] == 0
        assert sig[4] == 0

    def test_class_and_def_high_signal(self):
        lines = [
            "import os",
            "",
            "class DataProcessor:",
            "    def __init__(self, config):",
            "        pass",
            "    def process(self, data):",
            "        return data",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ]
        sig = compute_signal(lines, PY)
        assert sig[0] > 0.5
        assert sig[0] <= 2.0
        assert sig[1] == 0
        assert 1.0 <= sig[2] <= 2.0
        assert sig[3] > 0.9
        assert sig[4] < 0.5
        assert sig[5] > 0.9
        assert sig[6] > 0.1
        assert sig[7] == 0
        assert sig[8] > 0.2

    def test_decorators(self):
        sig = compute_signal(["@staticmethod", "def helper():", "    pass"], PY)
        assert 0.4 < sig[0] <= 2.0

    def test_indentation_increases_signal(self):
        sig = compute_signal(["pass", "    pass", "        pass", "            pass"], PY)
        assert sig[0] < sig[1] < sig[2] < sig[3]

    def test_keyword_touching_parens(self):
        sig = compute_signal(["for(x in y):", "if(cond):", "while(True):"], PY)
        assert all(s >= 0.3 for s in sig)

    def test_async_def_not_exceed_class(self):
        class_sig = compute_signal(["class Foo:"], PY)[0]
        async_sig = compute_signal(["async def foo():"], PY)[0]
        assert async_sig <= class_sig

    def test_member_access_no_keyword_leak(self):
        assert compute_signal(["x = obj.class"], PY)[0] == 0


class TestTypeScript:
    def test_zero_for_blank_and_comments(self):
        lines = ["", "  ", "// this is a comment", "   // indented comment"]
        assert compute_signal(lines, TS) == [0, 0, 0, 0]

    def test_zero_for_block_comments(self):
        lines = ["/* block comment */", "/* start", "middle", "end */", "const x = 1;"]
        sig = compute_signal(lines, TS)
        assert sig[0] == 0
        assert sig[1] == 0
        assert sig[2] == 0
        assert sig[3] == 0
        assert sig[4] > 0

    def test_detects_class_function_interface_enum(self):
        lines = [
            "import { readFile } from 'fs';",
            "",
            "export class Service {",
            "  constructor(private config: Config) {}",
            "",
            "  public async process(data: unknown): Promise<Result> {",
            "    const result = await this.transform(data);",
            "    return result;",
            "  }",
            "}",
            "",
            "export interface Config {",
            "  host: string;",
            "}",
            "",
            "export enum Status {",
            "  Active,",
            "  Inactive,",
            "}",
        ]
        sig = compute_signal(lines, TS)
        assert sig[0] > 0.5
        assert sig[2] >= 1.0
        assert sig[5] > 0.5
        assert sig[11] >= 0.9
        assert sig[15] >= 0.8

    def test_keyword_adjacent_punctuation(self):
        lines = ["export function foo(", "export class Bar{", "if(!ready)", "for(let x=0; x<10; x++)"]
        sig = compute_signal(lines, TS)
        assert sig[0] >= 1.0
        assert sig[1] >= 1.0
        assert sig[2] > 0.2
        assert sig[3] > 0.2

    def test_string_literal_block_comment_immunity(self):
        sig = compute_signal(['const s = "/* not a real comment */";', "const x = 1;"], TS)
        assert sig[1] > 0

    def test_url_string_not_stripped(self):
        sig = compute_signal(['export const url = "https://example.com";'], TS)
        assert sig[0] > 0.5

    def test_template_literal_block_comment_immunity(self):
        sig = compute_signal(["const t = `/* nope */`;", "function next() {}"], TS)
        assert sig[1] > 0.5

    def test_member_access_no_keyword_leak(self):
        assert compute_signal(["return obj.def;"], TS)[0] < 0.3

    def test_object_prototype_keys_no_nan(self):
        lines = [
            "  constructor(filename: string, content: string) {",
            "  toString() {",
            "  hasOwnProperty(k: string) {",
            "  valueOf() {",
        ]
        for s in compute_signal(lines, TS):
            assert not math.isnan(s)
            assert math.isfinite(s)


class TestGo:
    def test_func_keyword(self):
        sig = compute_signal(["func main() {", '  fmt.Println("hello")', "}"], GO)
        assert sig[0] >= 0.9

    def test_tab_indentation_matches_spaces(self):
        tab = compute_signal(['\t\tfmt.Println("x")'], GO)[0]
        space = compute_signal(['        fmt.Println("x")'], GO)[0]
        assert tab == pytest.approx(space, abs=1e-5)


class TestRange:
    def test_all_within_zero_two(self):
        lines = ["    " * 30 + "class Foo:" for _ in range(100)]
        for s in compute_signal(lines, PY):
            assert 0 <= s <= 2.0


class TestRustClojureQuote:
    def test_rust_lifetime_not_masking(self):
        rust = detect_language("test.rs")
        sig = compute_signal(["impl<'a> Foo for Bar<'a> {"], rust)
        assert sig[0] >= 1.0

    def test_clojure_quote_not_masking(self):
        clj = detect_language("test.clj")
        sig = compute_signal(["(def things '[a b c]) (defn realfn [] 1)"], clj)
        assert sig[0] >= 0.9


class TestPHP:
    def test_attribute_not_comment(self):
        php = detect_language("test.php")
        assert compute_signal(["#[Route('/x')]", "function handler() {}"], php)[0] > 0

    def test_plain_hash_is_comment(self):
        php = detect_language("test.php")
        assert compute_signal(["# real comment"], php)[0] == 0


class TestLanguageDetection:
    def test_js_vs_ts(self):
        assert detect_language("foo.js").name == "javascript"
        assert detect_language("foo.jsx").name == "javascript"
        assert detect_language("foo.mjs").name == "javascript"
        assert detect_language("foo.cjs").name == "javascript"
        assert detect_language("foo.ts").name == "typescript"
        assert detect_language("foo.tsx").name == "typescript"

    def test_js_lacks_ts_keywords(self):
        js = detect_language("foo.js")
        assert "interface" not in js.structural_keywords
        assert "enum" not in js.structural_keywords

    def test_edn_is_clojure_not_generic(self):
        assert detect_language("foo.edn").name != "generic"

    def test_unknown_ext_generic_fallthrough(self):
        lang = detect_language("foo.unknownext")
        assert ";" not in lang.comment_prefixes
        assert "//" not in lang.comment_prefixes

    def test_pyi_and_ruby_filenames(self):
        assert detect_language("stubs.pyi").name == "python"
        assert detect_language("Rakefile").name == "ruby"
        assert detect_language("Gemfile").name == "ruby"


class TestClojureForms:
    def test_recognizes_structural_forms(self):
        clj = detect_language("test.clj")
        lines = [
            "(defmulti area :shape)",
            "(defonce server (start))",
            "(letfn [(helper [x] x)] ...)",
            "(reify Foo (bar [_] 1))",
            "(extend-type String Foo (bar [_] 1))",
            "(extend-protocol Foo String (bar [_] 1))",
        ]
        for s in compute_signal(lines, clj):
            assert s > 0.3


class TestJavaAnnotations:
    def test_inline_annotation_raises_signal(self):
        java = detect_language("test.java")
        a = compute_signal(["public @Nullable String foo() {}"], java)[0]
        b = compute_signal(["public String foo() {}"], java)[0]
        assert a > b


class TestCommentTerminatorMasking:
    """Regressions for the terminator-swallowing class: string masking must
    never run over comment interiors (an apostrophe in prose swallowed the
    close delimiter and zeroed the rest of the file's signal)."""

    def test_block_comment_apostrophe_does_not_swallow_terminator(self):
        from wavelet.language import detect_language
        from wavelet.signal import compute_signal
        js = detect_language("a.js")
        sig = compute_signal(
            ["/* Don't use this directly */", "", "function go() {", "}"], js)
        assert sig[0] == 0.0
        assert sig[2] > 0.0, "function after a prose comment must score"

    def test_docstring_closer_with_apostrophe(self):
        from wavelet.language import detect_language
        from wavelet.signal import compute_signal
        py = detect_language("a.py")
        sig = compute_signal(
            ['def f():', '    """Doc.', "    Returns the user's data.\"\"\"",
             '    return 1', '', 'def g():', '    pass'], py)
        assert sig[5] > 0.0, "def g must score after the docstring closes"

    def test_hash_comment_with_odd_triple_quotes(self):
        from wavelet.language import detect_language
        from wavelet.signal import compute_signal
        py = detect_language("a.py")
        sig = compute_signal(['# see """ usage', 'def foo():', '    return 1'], py)
        assert sig[1] > 0.0, "a comment must not flip the docstring state"

    def test_python_backtick_is_not_a_string_delimiter(self):
        from wavelet.language import detect_language
        from wavelet.signal import compute_signal
        py = detect_language("a.py")
        sig = compute_signal(['x = 1  # `code` sample', 'def h():', '    pass'], py)
        assert sig[1] > 0.0

    def test_string_containing_comment_start_does_not_open_comment(self):
        from wavelet.language import detect_language
        from wavelet.signal import compute_signal
        js = detect_language("a.js")
        sig = compute_signal(
            ['var s = "/* not a comment */";', 'function ok() {', '}'], js)
        assert sig[1] > 0.0
