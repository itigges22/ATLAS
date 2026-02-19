# Contributing to ATLAS

Thank you for your interest in contributing to ATLAS! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

Before opening an issue:

1. Search existing issues to avoid duplicates
2. Check the [troubleshooting guide](docs/TROUBLESHOOTING.md)
3. Gather relevant information:
   - ATLAS version/commit hash
   - Operating system and version
   - GPU model and driver version
   - Output of `kubectl get pods`
   - Relevant logs

When opening an issue:

1. Use a clear, descriptive title
2. Describe the expected vs actual behavior
3. Provide steps to reproduce
4. Include logs and configuration (remove secrets!)

### Suggesting Features

Feature requests are welcome! Please:

1. Describe the use case and problem you're solving
2. Explain how your feature would work
3. Consider implementation complexity and trade-offs

### Pull Requests

#### Before You Start

1. Check existing issues/PRs for similar changes
2. For major changes, open an issue first to discuss
3. Fork the repository and create a feature branch

#### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/atlas.git
cd atlas

# Create feature branch
git checkout -b feature/your-feature-name

# Copy configuration
cp atlas.conf.example atlas.conf
# Edit atlas.conf for your environment

# Run tests before making changes
python tests/validate_tests.py
```

#### Making Changes

1. Follow existing code style and patterns
2. Write tests for new functionality
3. Update documentation as needed
4. Keep commits focused and atomic
5. Write clear commit messages

#### Commit Message Format

```
component: short description (50 chars max)

Longer description if needed. Explain what and why,
not how (the code shows how).

Fixes #123
```

Examples:
- `rag-api: add project caching for faster queries`
- `sandbox: increase default timeout to 90s`
- `docs: add GPU troubleshooting section`

#### Submitting

1. Ensure all tests pass: `python tests/validate_tests.py`
2. Update CHANGELOG.md if applicable
3. Push to your fork
4. Open a pull request with:
   - Clear title and description
   - Link to related issues
   - Test results

#### Code Review

- Address review feedback promptly
- Explain your decisions when disagreeing
- Request re-review after making changes

## Code Style

### Python

- Follow PEP 8
- Use type hints for function signatures
- Document public functions with docstrings
- Maximum line length: 100 characters

```python
def process_chunk(
    content: str,
    file_path: str,
    start_line: int,
) -> dict[str, Any]:
    """
    Process a code chunk for vector storage.

    Args:
        content: The chunk text content
        file_path: Source file path
        start_line: Starting line number

    Returns:
        Dictionary with chunk metadata and embedding
    """
    ...
```

### Bash

- Use shellcheck for linting
- Quote variables: `"$var"` not `$var`
- Use `[[` for conditionals
- Add comments for non-obvious logic

```bash
#!/bin/bash
set -euo pipefail

# Check if model file exists
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model not found at $MODEL_PATH" >&2
    exit 1
fi
```

### YAML/Kubernetes

- Use 2-space indentation
- Add resource limits to all containers
- Use meaningful names and labels

### Documentation

- Use Markdown for all documentation
- Include code examples where helpful
- Keep language clear and concise
- Update table of contents when adding sections

## Testing

### Running Tests

```bash
# Run all tests
python tests/validate_tests.py

# Run specific test file
pytest tests/test_rag_api.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names
- Test edge cases and error conditions

```python
def test_chunk_overlap_preserves_context():
    """Verify chunk overlap includes surrounding lines."""
    chunks = chunk_file(content, chunk_size=100, overlap=20)

    # Verify overlap exists
    assert chunks[0].end_line >= chunks[1].start_line
```

### Test Requirements

- New features must include tests
- Bug fixes should include regression tests
- Maintain or improve test coverage

## Architecture Decisions

When proposing architectural changes:

1. Document the problem and proposed solution
2. List alternatives considered
3. Explain trade-offs
4. Consider backwards compatibility
5. Update architecture documentation

## Release Process

Releases are handled by maintainers:

1. Update version in relevant files
2. Update CHANGELOG.md
3. Create git tag
4. Build and push container images
5. Create GitHub release

## License

This project is licensed under the **ATLAS Source Available License v1.0** (see [LICENSE](LICENSE)).

By submitting a contribution (pull request, patch, or any other form), you agree to the following terms:

- Your contributions are accepted under the same license as the project: the ATLAS Source Available License v1.0.
- You retain copyright of your contributions.
- You grant the project maintainer (Isaac Tigges) a perpetual, irrevocable, worldwide, royalty-free license to use, modify, and distribute your contributions under the project license.
- You represent that you have the legal right to grant this license and that your contributions do not infringe on any third-party rights.

## Questions?

- Check existing documentation
- Search closed issues
- Open a discussion for general questions

Thank you for contributing!
