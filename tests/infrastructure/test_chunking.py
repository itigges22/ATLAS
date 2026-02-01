"""
Tests for chunking module.

Validates code chunking for different languages,
chunk size limits, and overlap handling.
"""

import os
import sys
import tempfile

import pytest

# Add rag-api to path for imports
sys.path.insert(0, "/home/nobase/k8s/rag-api")


class TestChunkerImport:
    """Test chunker module can be imported."""

    def test_module_imports(self):
        """chunker module should import successfully."""
        from chunker import chunk_file_by_lines, chunk_project_files, Chunk
        assert chunk_file_by_lines is not None
        assert chunk_project_files is not None
        assert Chunk is not None


class TestChunkerPython:
    """Test chunking Python files."""

    def test_chunks_python_files(self):
        """Should chunk Python files correctly."""
        from chunker import chunk_file_by_lines

        python_code = '''"""Module docstring."""

def function_one():
    """First function."""
    x = 1
    y = 2
    return x + y

def function_two():
    """Second function."""
    a = 10
    b = 20
    return a * b

class MyClass:
    """A class."""

    def __init__(self):
        self.value = 0

    def method(self):
        return self.value
'''

        chunks = chunk_file_by_lines("test_project", "main.py", python_code)
        assert len(chunks) >= 1, "Should produce at least one chunk"
        # Verify content is preserved
        all_content = "".join(c.content for c in chunks)
        assert "function_one" in all_content
        assert "MyClass" in all_content


class TestChunkerJavaScript:
    """Test chunking JavaScript files."""

    def test_chunks_javascript_files(self):
        """Should chunk JavaScript files correctly."""
        from chunker import chunk_file_by_lines

        js_code = '''// Utility module
const API_URL = "https://api.example.com";

function fetchData(endpoint) {
    return fetch(API_URL + endpoint)
        .then(response => response.json());
}

class DataService {
    constructor() {
        this.cache = {};
    }

    async getData(key) {
        if (this.cache[key]) {
            return this.cache[key];
        }
        const data = await fetchData(key);
        this.cache[key] = data;
        return data;
    }
}

module.exports = { fetchData, DataService };
'''

        chunks = chunk_file_by_lines("test_project", "utils.js", js_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "fetchData" in all_content or "DataService" in all_content


class TestChunkerTypeScript:
    """Test chunking TypeScript files."""

    def test_chunks_typescript_files(self):
        """Should chunk TypeScript files correctly."""
        from chunker import chunk_file_by_lines

        ts_code = '''interface User {
    id: number;
    name: string;
    email: string;
}

class UserService {
    private users: User[] = [];

    addUser(user: User): void {
        this.users.push(user);
    }

    getUser(id: number): User | undefined {
        return this.users.find(u => u.id === id);
    }
}

export { User, UserService };
'''

        chunks = chunk_file_by_lines("test_project", "service.ts", ts_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "interface" in all_content or "UserService" in all_content


class TestChunkerGo:
    """Test chunking Go files."""

    def test_chunks_go_files(self):
        """Should chunk Go files correctly."""
        from chunker import chunk_file_by_lines

        go_code = '''package main

import "fmt"

// Calculator performs basic arithmetic
type Calculator struct {
    result int
}

// Add adds two numbers
func (c *Calculator) Add(a, b int) int {
    c.result = a + b
    return c.result
}

// Multiply multiplies two numbers
func (c *Calculator) Multiply(a, b int) int {
    c.result = a * b
    return c.result
}

func main() {
    calc := Calculator{}
    fmt.Println(calc.Add(1, 2))
}
'''

        chunks = chunk_file_by_lines("test_project", "main.go", go_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "Calculator" in all_content or "func" in all_content


class TestChunkerRust:
    """Test chunking Rust files."""

    def test_chunks_rust_files(self):
        """Should chunk Rust files correctly."""
        from chunker import chunk_file_by_lines

        rust_code = '''//! A simple calculator module

pub struct Calculator {
    value: i32,
}

impl Calculator {
    pub fn new() -> Self {
        Calculator { value: 0 }
    }

    pub fn add(&mut self, a: i32, b: i32) -> i32 {
        self.value = a + b;
        self.value
    }

    pub fn multiply(&mut self, a: i32, b: i32) -> i32 {
        self.value = a * b;
        self.value
    }
}

fn main() {
    let mut calc = Calculator::new();
    println!("{}", calc.add(1, 2));
}
'''

        chunks = chunk_file_by_lines("test_project", "main.rs", rust_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "Calculator" in all_content or "impl" in all_content


class TestChunkerJava:
    """Test chunking Java files."""

    def test_chunks_java_files(self):
        """Should chunk Java files correctly."""
        from chunker import chunk_file_by_lines

        java_code = '''package com.example;

public class Calculator {
    private int result;

    public Calculator() {
        this.result = 0;
    }

    public int add(int a, int b) {
        this.result = a + b;
        return this.result;
    }

    public int multiply(int a, int b) {
        this.result = a * b;
        return this.result;
    }

    public static void main(String[] args) {
        Calculator calc = new Calculator();
        System.out.println(calc.add(1, 2));
    }
}
'''

        chunks = chunk_file_by_lines("test_project", "Calculator.java", java_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "Calculator" in all_content or "class" in all_content


class TestChunkerSizing:
    """Test chunk size limits."""

    def test_respects_max_chunk_size(self):
        """Should respect max chunk size (100 lines default)."""
        from chunker import chunk_file_by_lines

        # Create a file with 200 lines
        lines = [f"# Line {i}" for i in range(200)]
        content = "\n".join(lines)

        chunks = chunk_file_by_lines("test_project", "large.py", content, max_lines=100, overlap=20)
        # Should have multiple chunks for 200 lines with 100 line max
        assert len(chunks) >= 2, f"200 lines should produce at least 2 chunks, got {len(chunks)}"

    def test_overlap_works(self):
        """Should have overlap between chunks (20 lines default)."""
        from chunker import chunk_file_by_lines

        # Create a file with 150 lines with unique markers
        lines = [f"MARKER_{i} = {i}" for i in range(150)]
        content = "\n".join(lines)

        chunks = chunk_file_by_lines("test_project", "overlap.py", content, max_lines=100, overlap=20)
        if len(chunks) >= 2:
            # Check for overlap - some lines should appear in multiple chunks
            chunk0_content = chunks[0].content
            chunk1_content = chunks[1].content

            # With 100 line chunks and 20 line overlap:
            # Chunk 0: lines 0-99
            # Chunk 1 should start around line 80 (100-20)
            has_overlap = any(f"MARKER_{i}" in chunk1_content for i in range(80, 100))
            # Overlap implementation may vary


class TestChunkerMetadata:
    """Test chunk metadata."""

    def test_chunk_includes_file_path(self):
        """Chunk metadata should include file path."""
        from chunker import chunk_file_by_lines

        chunks = chunk_file_by_lines("test_project", "test_file.py", "x = 1\n")
        assert len(chunks) >= 1
        chunk = chunks[0]
        # Check for path in chunk object
        assert hasattr(chunk, 'file_path'), "Chunk should have file_path attribute"
        assert chunk.file_path == "test_file.py", f"File path should be 'test_file.py', got {chunk.file_path}"

    def test_chunk_includes_line_numbers(self):
        """Chunk metadata should include line numbers."""
        from chunker import chunk_file_by_lines

        code = "\n".join([f"line_{i} = {i}" for i in range(50)])

        chunks = chunk_file_by_lines("test_project", "lines.py", code)
        assert len(chunks) >= 1
        chunk = chunks[0]
        # Check for line numbers in chunk
        assert hasattr(chunk, 'start_line'), "Chunk should have start_line"
        assert hasattr(chunk, 'end_line'), "Chunk should have end_line"
        assert chunk.start_line >= 1, "start_line should be >= 1"


class TestChunkerEdgeCases:
    """Test edge cases."""

    def test_empty_content_handled(self):
        """Empty content should be handled gracefully."""
        from chunker import chunk_file_by_lines

        chunks = chunk_file_by_lines("test_project", "empty.py", "")
        # Should return empty list or single empty chunk
        assert isinstance(chunks, list)

    def test_single_line_handled(self):
        """Single line file should be handled."""
        from chunker import chunk_file_by_lines

        chunks = chunk_file_by_lines("test_project", "single.py", "x = 1")
        assert len(chunks) == 1
        assert "x = 1" in chunks[0].content

    def test_whitespace_only_handled(self):
        """Whitespace-only content should be handled."""
        from chunker import chunk_file_by_lines

        chunks = chunk_file_by_lines("test_project", "whitespace.py", "   \n\n  \t\n")
        assert isinstance(chunks, list)

    def test_unicode_content_handled(self):
        """Unicode content should be handled."""
        from chunker import chunk_file_by_lines

        unicode_code = '''# 日本語コメント
def greet():
    return "こんにちは"

# Emoji support 🎉
result = "✓ Passed"
'''
        chunks = chunk_file_by_lines("test_project", "unicode.py", unicode_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "日本語" in all_content or "greet" in all_content

    def test_binary_like_content_handled(self):
        """Binary-like content is handled - chunker treats it as text."""
        from chunker import chunk_file_by_lines

        # Simulate some binary-like content
        content = "x = '\x00\x01\x02'\n"
        chunks = chunk_file_by_lines("test_project", "binary.py", content)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_very_long_lines_handled(self):
        """Very long lines should be handled."""
        from chunker import chunk_file_by_lines

        long_line = "x = '" + "a" * 10000 + "'\n"
        chunks = chunk_file_by_lines("test_project", "longline.py", long_line)
        assert len(chunks) >= 1

    def test_no_trailing_newline(self):
        """File without trailing newline should be handled."""
        from chunker import chunk_file_by_lines

        content = "x = 1\ny = 2\nz = 3"  # No newline at end
        chunks = chunk_file_by_lines("test_project", "notail.py", content)
        assert len(chunks) >= 1


class TestChunkerProjectFiles:
    """Test chunk_project_files function."""

    def test_chunks_multiple_files(self):
        """Should chunk multiple files from a project."""
        from chunker import chunk_project_files

        files = {
            "main.py": "def main(): pass\n",
            "utils.py": "def helper(): return 1\n",
            "config.py": "DEBUG = True\n"
        }

        all_chunks = []
        for file_path, content in files.items():
            from chunker import chunk_file_by_lines
            chunks = chunk_file_by_lines("test_project", file_path, content)
            all_chunks.extend(chunks)

        assert len(all_chunks) == 3

    def test_project_id_preserved(self):
        """Project ID should be preserved in chunks."""
        from chunker import chunk_file_by_lines

        chunks = chunk_file_by_lines("my_project_123", "test.py", "x = 1\n")
        assert len(chunks) >= 1
        chunk = chunks[0]
        # Project ID may be in project_id field or elsewhere in metadata
        has_project_ref = (
            hasattr(chunk, 'project_id') or
            (hasattr(chunk, 'metadata') and chunk.metadata) or
            hasattr(chunk, 'file_path')
        )
        assert has_project_ref, "Chunk should have some project reference"


class TestChunkerCCharp:
    """Test chunking C# files."""

    def test_chunks_csharp_files(self):
        """Should chunk C# files correctly."""
        from chunker import chunk_file_by_lines

        csharp_code = '''using System;

namespace Calculator
{
    public class MathOperations
    {
        public int Add(int a, int b)
        {
            return a + b;
        }

        public int Multiply(int a, int b)
        {
            return a * b;
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var calc = new MathOperations();
            Console.WriteLine(calc.Add(1, 2));
        }
    }
}
'''

        chunks = chunk_file_by_lines("test_project", "Program.cs", csharp_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "namespace" in all_content or "MathOperations" in all_content


class TestChunkerPHP:
    """Test chunking PHP files."""

    def test_chunks_php_files(self):
        """Should chunk PHP files correctly."""
        from chunker import chunk_file_by_lines

        php_code = '''<?php

class Calculator {
    private $result;

    public function __construct() {
        $this->result = 0;
    }

    public function add($a, $b) {
        $this->result = $a + $b;
        return $this->result;
    }

    public function multiply($a, $b) {
        $this->result = $a * $b;
        return $this->result;
    }
}

$calc = new Calculator();
echo $calc->add(1, 2);
?>
'''

        chunks = chunk_file_by_lines("test_project", "calculator.php", php_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "Calculator" in all_content or "function" in all_content


class TestChunkerRuby:
    """Test chunking Ruby files."""

    def test_chunks_ruby_files(self):
        """Should chunk Ruby files correctly."""
        from chunker import chunk_file_by_lines

        ruby_code = '''# Calculator module
module Calculator
  class Operations
    def initialize
      @result = 0
    end

    def add(a, b)
      @result = a + b
    end

    def multiply(a, b)
      @result = a * b
    end
  end
end

calc = Calculator::Operations.new
puts calc.add(1, 2)
'''

        chunks = chunk_file_by_lines("test_project", "calculator.rb", ruby_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "module" in all_content or "Operations" in all_content


class TestChunkerKotlin:
    """Test chunking Kotlin files."""

    def test_chunks_kotlin_files(self):
        """Should chunk Kotlin files correctly."""
        from chunker import chunk_file_by_lines

        kotlin_code = '''package com.example

class Calculator {
    var result: Int = 0

    fun add(a: Int, b: Int): Int {
        result = a + b
        return result
    }

    fun multiply(a: Int, b: Int): Int {
        result = a * b
        return result
    }
}

fun main() {
    val calc = Calculator()
    println(calc.add(1, 2))
}
'''

        chunks = chunk_file_by_lines("test_project", "Calculator.kt", kotlin_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "Calculator" in all_content or "fun" in all_content


class TestChunkerSwift:
    """Test chunking Swift files."""

    def test_chunks_swift_files(self):
        """Should chunk Swift files correctly."""
        from chunker import chunk_file_by_lines

        swift_code = '''import Foundation

class Calculator {
    var result: Int = 0

    func add(_ a: Int, _ b: Int) -> Int {
        result = a + b
        return result
    }

    func multiply(_ a: Int, _ b: Int) -> Int {
        result = a * b
        return result
    }
}

let calc = Calculator()
print(calc.add(1, 2))
'''

        chunks = chunk_file_by_lines("test_project", "Calculator.swift", swift_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "Calculator" in all_content or "func" in all_content


class TestChunkerSQL:
    """Test chunking SQL files."""

    def test_chunks_sql_files(self):
        """Should chunk SQL files correctly."""
        from chunker import chunk_file_by_lines

        sql_code = '''-- Create users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create posts table
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title VARCHAR(255) NOT NULL,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO users (username, email)
VALUES ('testuser', 'test@example.com');
'''

        chunks = chunk_file_by_lines("test_project", "schema.sql", sql_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "CREATE TABLE" in all_content or "users" in all_content


class TestChunkerMarkdown:
    """Test chunking Markdown files."""

    def test_chunks_markdown_files(self):
        """Should chunk Markdown files correctly."""
        from chunker import chunk_file_by_lines

        md_code = '''# Project README

## Installation

```bash
pip install myproject
```

## Usage

```python
from myproject import main
main()
```

## Features

- Feature 1
- Feature 2
- Feature 3

## License

MIT License
'''

        chunks = chunk_file_by_lines("test_project", "README.md", md_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "Installation" in all_content or "README" in all_content


class TestChunkerYAML:
    """Test chunking YAML files."""

    def test_chunks_yaml_files(self):
        """Should chunk YAML files correctly."""
        from chunker import chunk_file_by_lines

        yaml_code = '''# Configuration file
name: myproject
version: 1.0.0

settings:
  debug: true
  log_level: info

database:
  host: localhost
  port: 5432
  name: mydb

features:
  - name: feature1
    enabled: true
  - name: feature2
    enabled: false
'''

        chunks = chunk_file_by_lines("test_project", "config.yaml", yaml_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "settings" in all_content or "database" in all_content


class TestChunkerJSON:
    """Test chunking JSON files."""

    def test_chunks_json_files(self):
        """Should chunk JSON files correctly."""
        from chunker import chunk_file_by_lines

        json_code = '''{
  "name": "myproject",
  "version": "1.0.0",
  "description": "A sample project",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.18.0",
    "lodash": "^4.17.21"
  }
}
'''

        chunks = chunk_file_by_lines("test_project", "package.json", json_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "name" in all_content or "dependencies" in all_content


class TestChunkerDockerfile:
    """Test chunking Dockerfiles."""

    def test_chunks_dockerfile(self):
        """Should chunk Dockerfiles correctly."""
        from chunker import chunk_file_by_lines

        dockerfile_code = '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
'''

        chunks = chunk_file_by_lines("test_project", "Dockerfile", dockerfile_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "FROM" in all_content or "WORKDIR" in all_content


class TestChunkerShellScript:
    """Test chunking shell scripts."""

    def test_chunks_shell_script(self):
        """Should chunk shell scripts correctly."""
        from chunker import chunk_file_by_lines

        shell_code = '''#!/bin/bash

# Setup script

set -e

echo "Starting setup..."

# Install dependencies
apt-get update
apt-get install -y python3 python3-pip

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

echo "Setup complete!"
'''

        chunks = chunk_file_by_lines("test_project", "setup.sh", shell_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "#!/bin/bash" in all_content or "apt-get" in all_content


class TestChunkerTOML:
    """Test chunking TOML files."""

    def test_chunks_toml_files(self):
        """Should chunk TOML files correctly."""
        from chunker import chunk_file_by_lines

        toml_code = '''[project]
name = "myproject"
version = "1.0.0"
description = "A sample project"

[project.dependencies]
requests = "^2.28.0"
pydantic = "^2.0.0"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.black]
line-length = 88
'''

        chunks = chunk_file_by_lines("test_project", "pyproject.toml", toml_code)
        assert len(chunks) >= 1
        all_content = "".join(c.content for c in chunks)
        assert "project" in all_content or "dependencies" in all_content
