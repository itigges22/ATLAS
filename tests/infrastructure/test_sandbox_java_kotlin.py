"""
Tests for Java (and later Kotlin) sandbox executor support.

Validates code execution, syntax checking, class name extraction,
compile errors, and runtime errors for JVM languages.

All Java tests are gated with skipif(javac is None) so CI runners
without a JDK don't fail — the executor boots on the host runner
which has no JDK installed.
"""

import pytest
import shutil

# importorskip, not a plain import — see test_llm.py: keeps collection
# alive on environments without the integration deps.
httpx = pytest.importorskip("httpx")

# Reusable skipif marker for all classes that need javac.
_requires_javac = pytest.mark.skipif(
    shutil.which("javac") is None,
    reason="javac not available",
)


@_requires_javac
class TestJavaExecution:
    """Test Java code execution in sandbox."""

    def test_hello_world(self, sandbox_client: httpx.Client):
        """Basic javac → java pipeline: compile and run."""
        code = """\
public class Main {
    public static void main(String[] args) {
        System.out.println("hello world");
    }
}
"""
        response = sandbox_client.post(
            "/execute",
            json={"code": code, "language": "java"},
            timeout=60.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True, f"Hello world should succeed: {data}"
        assert data.get("compile_success") is True
        assert "hello world" in data.get("stdout", "")

    def test_computed_values(self, sandbox_client: httpx.Client):
        """Code that computes values should capture output."""
        code = """\
public class Main {
    public static void main(String[] args) {
        int a = 2, b = 3;
        System.out.println("Result: " + (a + b));
    }
}
"""
        response = sandbox_client.post(
            "/execute",
            json={"code": code, "language": "java"},
            timeout=60.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
        assert "Result: 5" in data.get("stdout", "")

    def test_custom_class_name(self, sandbox_client: httpx.Client):
        """Public class name extraction: file must be Calculator.java."""
        code = """\
public class Calculator {
    public static void main(String[] args) {
        System.out.println("sum=" + (10 + 20));
    }
}
"""
        response = sandbox_client.post(
            "/execute",
            json={"code": code, "language": "java"},
            timeout=60.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True, (
            f"Custom class name should work: {data}"
        )
        assert "sum=30" in data.get("stdout", "")

    def test_compile_error(self, sandbox_client: httpx.Client):
        """Missing semicolon should fail compilation."""
        code = """\
public class Main {
    public static void main(String[] args) {
        int x = 1
        System.out.println(x);
    }
}
"""
        response = sandbox_client.post(
            "/execute",
            json={"code": code, "language": "java"},
            timeout=60.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("compile_success") is False
        assert data.get("success") is False
        error_msg = data.get("stderr", "") + data.get("error_message", "")
        assert "error" in error_msg.lower(), (
            f"Compile error not reported: {error_msg}"
        )

    def test_runtime_error_npe(self, sandbox_client: httpx.Client):
        """NullPointerException should be caught as runtime error."""
        code = """\
public class Main {
    public static void main(String[] args) {
        String s = null;
        System.out.println(s.length());
    }
}
"""
        response = sandbox_client.post(
            "/execute",
            json={"code": code, "language": "java"},
            timeout=60.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("compile_success") is True
        assert data.get("success") is False
        error_msg = data.get("stderr", "") + data.get("error_message", "")
        assert "NullPointerException" in error_msg, (
            f"NPE not reported: {error_msg}"
        )

    def test_runtime_error_division_by_zero(self, sandbox_client: httpx.Client):
        """ArithmeticException from integer division by zero."""
        code = """\
public class Main {
    public static void main(String[] args) {
        int x = 1 / 0;
    }
}
"""
        response = sandbox_client.post(
            "/execute",
            json={"code": code, "language": "java"},
            timeout=60.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is False
        error_msg = data.get("stderr", "") + data.get("error_message", "")
        assert (
            "ArithmeticException" in error_msg
            or "/ by zero" in error_msg
        )

    def test_import_nonexistent_package(self, sandbox_client: httpx.Client):
        """Importing a package that does not exist should fail compilation."""
        code = """\
import xyz.NonExistent;
public class Main {
    public static void main(String[] args) {
        NonExistent obj = new NonExistent();
    }
}
"""
        response = sandbox_client.post(
            "/execute",
            json={"code": code, "language": "java"},
            timeout=60.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is False
        error_msg = data.get("stderr", "") + data.get("error_message", "")
        assert (
            "package" in error_msg.lower()
            or "does not exist" in error_msg.lower()
        ), f"Error should mention missing package: {error_msg}"


@_requires_javac
class TestJavaSyntaxCheck:
    """Test /syntax-check endpoint for Java."""

    def test_valid_java(self, sandbox_client: httpx.Client):
        """Well-formed Java should pass syntax check."""
        code = """\
public class Main {
    public static void main(String[] args) {
        System.out.println("ok");
    }
}
"""
        response = sandbox_client.post(
            "/syntax-check",
            json={"code": code, "language": "java"},
            timeout=60.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("valid") is True, f"Valid Java rejected: {data}"
        assert data.get("errors") == [] or data.get("errors") is None

    def test_invalid_java(self, sandbox_client: httpx.Client):
        """Broken Java should fail syntax check with error details."""
        code = """\
public class Main {
    public static void main(String[] args) {
        int x = 1
    }
}
"""
        response = sandbox_client.post(
            "/syntax-check",
            json={"code": code, "language": "java"},
            timeout=60.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("valid") is False
        errors = data.get("errors", [])
        assert len(errors) > 0, "Should report at least one error"


class TestJavaLanguagesEndpoint:
    """Test /languages reports Java — no skipif needed, endpoint
    returns 'not installed' gracefully when javac is absent."""

    def test_java_in_languages(self, sandbox_client: httpx.Client):
        """Java should appear in the /languages response."""
        response = sandbox_client.get("/languages")
        assert response.status_code == 200
        data = response.json()
        languages = data.get("languages", {})
        assert "java" in languages, (
            f"Java missing from /languages: {list(languages.keys())}"
        )
