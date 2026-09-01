from stages.llm_client import extract_code


def test_extract_code_accepts_non_python_language_fence():
    response = """Here is the file:
```javascript
const result = `${2 + 3}`;
```
"""

    assert extract_code(response) == "const result = `${2 + 3}`;\n"


def test_extract_code_accepts_punctuation_in_language_label():
    response = """```c++
int main() { return 0; }
```"""

    assert extract_code(response) == "int main() { return 0; }\n"


def test_extract_code_chooses_longest_fenced_block_across_languages():
    response = """```text
short
```
```typescript
export function add(a: number, b: number): number { return a + b; }
```"""

    assert extract_code(response).startswith("export function add")


# --- exact bytes -------------------------------------------------------------
#
# The extractor names the artifact. Every candidate hash, selection record,
# authorization identity and disk write downstream is computed from the bytes
# it returns, so the bytes inside the fence must come back exactly as the model
# wrote them: a final newline when there is one, none when there is not, and
# every trailing blank line in between. The Markdown fence is framing and is
# not part of the artifact; the artifact's own line terminator is.

import pytest

from stages.plan_search import extract_code_from_response
from stages.pr_cot import extract_code_from_repair

EXTRACTORS = [
    pytest.param(extract_code, id="llm_client.extract_code"),
    pytest.param(extract_code_from_response, id="plan_search.extract_code_from_response"),
    pytest.param(extract_code_from_repair, id="pr_cot.extract_code_from_repair"),
]


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_fenced_code_keeps_its_final_newline(extract):
    assert extract("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1\n"


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_fenced_code_without_a_final_newline_gets_none(extract):
    assert extract("```python\ndef f():\n    return 1```") == "def f():\n    return 1"


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_fenced_code_keeps_every_trailing_blank_line(extract):
    assert extract("```python\ndef f():\n    return 1\n\n\n```") == "def f():\n    return 1\n\n\n"


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_fenced_code_keeps_its_line_endings(extract):
    assert extract("```python\r\nx = 1\r\ny = 2\r\n```") == "x = 1\r\ny = 2\r\n"


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_inline_code_keeps_its_final_newline(extract):
    # No fence: the response is the artifact, and its terminator stays.
    assert extract("def f():\n    return 1\n") == "def f():\n    return 1\n"


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_inline_and_fenced_forms_share_bytes_only_when_the_bytes_agree(extract):
    fenced = extract("```\ndef f():\n    return 1\n```")
    inline = extract("def f():\n    return 1\n")
    bare = extract("def f():\n    return 1")
    assert fenced == inline
    assert bare != fenced
    assert bare + "\n" == fenced


def test_extract_code_prose_before_a_fence_is_not_part_of_the_artifact():
    response = "Here is the file:\n```javascript\nconst x = 1;\n```\nThat should do it.\n"
    assert extract_code(response) == "const x = 1;\n"
