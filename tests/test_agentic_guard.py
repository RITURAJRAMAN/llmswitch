"""Optional agentic extra: the core import stays free of it, and a missing dependency is reported clearly."""
import subprocess
import sys

import pytest


def test_importing_llmswitch_does_not_load_the_agentic_dependency():
    code = ("import sys, llmswitch; "
            "print(any(name.startswith(('langchain', 'langchain_core')) for name in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_the_agentic_package_imports_without_its_dependency():
    import llmswitch.agentic  # noqa: F401


def test_a_missing_dependency_gives_the_install_command(monkeypatch):
    # Hide every cached dependency module too: a cached submodule would otherwise import without its parent.
    for name in list(sys.modules):
        if name == "langchain_core" or name.startswith("langchain_core."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "llmswitch.agentic.chat", raising=False)
    monkeypatch.delitem(sys.modules, "llmswitch.agentic.messages", raising=False)
    with pytest.raises(ImportError, match=r'pip install "llmswitch-sdk\[agentic\]"'):
        import llmswitch.agentic.chat  # noqa: F401
