import os

from atlas import runtime_artifacts
from atlas.commands import tui


def _executable(path, content="binary"):
    path.write_text(content)
    path.chmod(0o755)


def test_go_binary_current_only_when_at_least_as_new_as_source(tmp_path):
    source = tmp_path / "tui"
    source.mkdir()
    main_go = source / "main.go"
    main_go.write_text("package main\n")
    binary = tmp_path / "atlas-tui"
    _executable(binary)

    os.utime(binary, ns=(100, 100))
    os.utime(main_go, ns=(200, 200))
    assert not runtime_artifacts.go_binary_is_current(str(binary), str(source))

    os.utime(binary, ns=(300, 300))
    assert runtime_artifacts.go_binary_is_current(str(binary), str(source))


def test_embedded_demo_prompt_change_invalidates_tui_binary(tmp_path):
    source = tmp_path / "tui"
    source.mkdir()
    (source / "main.go").write_text("package main\n")
    prompts = source / "demo_prompts_fallback.json"
    prompts.write_text("[]\n")
    binary = tmp_path / "atlas-tui"
    _executable(binary)

    os.utime(binary, ns=(200, 200))
    os.utime(prompts, ns=(300, 300))
    assert not runtime_artifacts.go_binary_is_current(
        str(binary), str(source), ("demo_prompts_fallback.json",)
    )


def test_select_tui_rebuilds_stale_binary(monkeypatch, tmp_path):
    atlas_dir = tmp_path / "atlas"
    source = atlas_dir / "tui"
    source.mkdir(parents=True)
    main_go = source / "main.go"
    main_go.write_text("package main\n")
    binary = tmp_path / "atlas-tui"
    _executable(binary)
    os.utime(binary, ns=(100, 100))
    os.utime(main_go, ns=(200, 200))

    monkeypatch.setattr(tui, "_find_tui_binary", lambda _: str(binary))
    monkeypatch.setattr(tui, "_build_tui", lambda _: "/rebuilt/atlas-tui")

    assert tui._select_tui_binary(str(atlas_dir)) == "/rebuilt/atlas-tui"


def test_select_tui_reuses_current_binary(monkeypatch, tmp_path):
    atlas_dir = tmp_path / "atlas"
    source = atlas_dir / "tui"
    source.mkdir(parents=True)
    main_go = source / "main.go"
    main_go.write_text("package main\n")
    binary = tmp_path / "atlas-tui"
    _executable(binary)
    os.utime(main_go, ns=(100, 100))
    os.utime(binary, ns=(200, 200))

    monkeypatch.setattr(tui, "_find_tui_binary", lambda _: str(binary))
    monkeypatch.setattr(
        tui, "_build_tui", lambda _: (_ for _ in ()).throw(AssertionError("rebuilt"))
    )

    assert tui._select_tui_binary(str(atlas_dir)) == str(binary)
