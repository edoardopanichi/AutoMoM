from __future__ import annotations

import backend.pipeline.compute as compute_module


def test_normalize_compute_preference() -> None:
    """! @brief Test normalize compute preference.
    """
    assert compute_module.normalize_compute_preference("auto") == "auto"
    assert compute_module.normalize_compute_preference("CUDA") == "cuda"
    assert compute_module.normalize_compute_preference("cpu") == "cpu"
    assert compute_module.normalize_compute_preference("invalid") == "auto"


def test_resolve_torch_device_auto_prefers_cuda(monkeypatch) -> None:
    """! @brief Test resolve torch device auto prefers cuda.
    @param monkeypatch Value for monkeypatch.
    """
    monkeypatch.setattr(compute_module, "torch_cuda_available", lambda _device_id=0: True)
    assert compute_module.resolve_torch_device("auto", 0) == "cuda"


def test_resolve_torch_device_forced_cuda_falls_back(monkeypatch) -> None:
    """! @brief Test resolve torch device forced cuda falls back.
    @param monkeypatch Value for monkeypatch.
    """
    monkeypatch.setattr(compute_module, "torch_cuda_available", lambda _device_id=0: False)
    assert compute_module.resolve_torch_device("cuda", 0) == "cpu"


def test_should_enable_native_gpu_respects_cpu_preference(monkeypatch) -> None:
    """! @brief Test should enable native gpu respects cpu preference.
    @param monkeypatch Value for monkeypatch.
    """
    monkeypatch.setattr(compute_module, "native_cuda_available", lambda _device_id=0: True)
    assert compute_module.should_enable_native_gpu("cpu", 0) is False

