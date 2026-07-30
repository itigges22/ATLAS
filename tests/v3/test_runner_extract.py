from stages.llm_client import extract_code


def test_extract_code_accepts_non_python_language_fence():
    response = """Here is the file:
```javascript
const result = `${2 + 3}`;
```
"""

    assert extract_code(response) == "const result = `${2 + 3}`;"


def test_extract_code_accepts_punctuation_in_language_label():
    response = """```c++
int main() { return 0; }
```"""

    assert extract_code(response) == "int main() { return 0; }"


def test_extract_code_chooses_longest_fenced_block_across_languages():
    response = """```text
short
```
```typescript
export function add(a: number, b: number): number { return a + b; }
```"""

    assert extract_code(response).startswith("export function add")
