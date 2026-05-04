import torch

from htrm.device import get_device, sync


def test_get_device_returns_torch_device():
    d = get_device()
    assert isinstance(d, torch.device)


def test_tensor_round_trip_to_device():
    d = get_device()
    x = torch.randn(4, 4)
    y = x.to(d) + 1.0
    z = y.cpu()
    assert torch.allclose(z, x + 1.0, atol=1e-4)


def test_sync_does_not_raise():
    # sync() is a no-op on devices that don't expose synchronize; just
    # confirm it never raises so calling code can use it unconditionally.
    sync()
