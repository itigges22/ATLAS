"""Runtime sizing invariants not tied to a particular model family."""

from atlas.cli.commands.fit import GIB, GGUFMeta, fit_runtime_knobs


def _model() -> GGUFMeta:
    return GGUFMeta(
        path="fixture.gguf", file_size=int(7 * GIB), architecture="fixture",
        n_layers=32, n_embd=4096, n_head=16, head_dim=256, v_len=256,
        k_len_swa=256, v_len_swa=256, kv_heads=[4] * 32,
        n_ctx_train=262144, sliding_window=0, local_mask=[False] * 32,
        swa_note="full attention",
    )


def test_fit_keeps_embedding_batch_equal_to_microbatch():
    result = fit_runtime_knobs(_model(), vram_gib=13, slots=1)
    assert result.fits is True
    assert result.ubatch == 1024
    assert result.batch == result.ubatch


def test_unfit_result_still_reports_runnable_batch_geometry():
    result = fit_runtime_knobs(_model(), vram_gib=10, slots=1)
    assert result.fits is False
    assert result.batch == result.ubatch == 512
