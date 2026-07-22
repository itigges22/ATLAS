import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name):
    path = ROOT / "asa_calibration" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, prompt):
        self.prompt = prompt

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"prompt": self.prompt}).encode()


def test_hidden_state_builder_uses_loaded_models_template(monkeypatch):
    module = _load("build_steering_vector.py")
    module._render_user_context.cache_clear()
    monkeypatch.setattr(module, "urlopen", lambda *_a, **_k: _Response("MODEL_TEMPLATE:"))
    rendered = module.render_prompt({
        "user": "edit a function",
        "assistant_prefix": '{"name":"structural_edit"',
    })
    assert rendered == 'MODEL_TEMPLATE:{"name":"structural_edit"'
    assert "<|im_start|>" not in rendered


def test_legacy_cvector_builder_uses_loaded_models_template(monkeypatch):
    module = _load("build_cvector_prompts.py")
    monkeypatch.setattr(module, "urlopen", lambda *_a, **_k: _Response("MODEL\nTEMPLATE\n"))
    rendered = module.render({
        "user": "edit a function",
        "assistant_prefix": '{"name":"structural_edit"',
    }, "http://llama-server:8080")
    assert rendered == 'MODEL\\nTEMPLATE\\n{"name":"structural_edit"'
    assert "<|im_start|>" not in rendered
