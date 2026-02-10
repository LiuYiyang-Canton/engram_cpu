"""RMSNorm dot-product module implemented for CPU float32 tensors."""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn


class RMSNormDotProduct(nn.Module):
    """Compute dot products of RMS-normalized query/key groups."""

    def __init__(self, eps: float = 1e-5) -> None:
        """
        Initialize the RMSNorm dot-product module.

        Main feature:
            Stores epsilon for numerically stable RMS normalization.

        Inputs:
            eps: float scalar epsilon added to RMS denominator; must be >= 0

        Outputs:
            None
        """
        super().__init__()
        if eps < 0:
            raise ValueError(f"`eps` must be >= 0, but got {eps}.")
        self.eps = float(eps)

    @staticmethod
    def _validate_inputs(h: Tensor, k: Tensor, gamma1: Tensor, gamma2: Tensor) -> None:
        """
        Validate tensor device, dtype, and shape constraints for forward pass.

        Main feature:
            Enforces CPU + float32 tensors and compatible shapes before compute.

        Inputs:
            h: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            k: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            gamma1: float32 tensor of shape [hc_mult, d] on CPU
            gamma2: float32 tensor of shape [hc_mult, d] on CPU

        Outputs:
            None
        """
        tensors = {"h": h, "k": k, "gamma1": gamma1, "gamma2": gamma2}
        for name, tensor in tensors.items():
            if tensor.device.type != "cpu":
                raise ValueError(f"`{name}` must be on CPU, but got device {tensor.device}.")
            if tensor.dtype != torch.float32:
                raise TypeError(f"`{name}` must be torch.float32, but got dtype {tensor.dtype}.")

        if h.ndim != 4 or k.ndim != 4:
            raise ValueError(
                f"`h` and `k` must be rank-4 tensors, but got ndim={h.ndim} and ndim={k.ndim}."
            )
        if h.shape != k.shape:
            raise ValueError(f"`h` and `k` must share shape, but got {h.shape} and {k.shape}.")

        batch_size, seq_len, hc_mult, d = h.shape

        if gamma1.ndim != 2 or gamma2.ndim != 2:
            raise ValueError(
                "`gamma1` and `gamma2` must be rank-2 tensors of shape [hc_mult, d], "
                f"but got ndim={gamma1.ndim} and ndim={gamma2.ndim}."
            )
        expected_gamma_shape = (hc_mult, d)
        if tuple(gamma1.shape) != expected_gamma_shape:
            raise ValueError(
                f"`gamma1` must have shape {expected_gamma_shape}, but got {tuple(gamma1.shape)}."
            )
        if tuple(gamma2.shape) != expected_gamma_shape:
            raise ValueError(
                f"`gamma2` must have shape {expected_gamma_shape}, but got {tuple(gamma2.shape)}."
            )

        if batch_size <= 0 or seq_len <= 0 or hc_mult <= 0 or d <= 0:
            raise ValueError(f"All dimensions must be positive, but got shape {tuple(h.shape)}.")

    @staticmethod
    def _validate_grad_out(grad_out: Tensor, expected_shape: tuple[int, int, int]) -> None:
        """
        Validate upstream gradient tensor for manual backward computation.

        Main feature:
            Enforces CPU + float32 and exact [batch_size, seq_len, hc_mult] shape.

        Inputs:
            grad_out: float32 tensor of shape [batch_size, seq_len, hc_mult] on CPU
            expected_shape: tuple[int, int, int] for [batch_size, seq_len, hc_mult]

        Outputs:
            None
        """
        if grad_out.device.type != "cpu":
            raise ValueError(f"`grad_out` must be on CPU, but got device {grad_out.device}.")
        if grad_out.dtype != torch.float32:
            raise TypeError(f"`grad_out` must be torch.float32, but got dtype {grad_out.dtype}.")
        if grad_out.ndim != 3:
            raise ValueError(f"`grad_out` must be rank-3, but got ndim={grad_out.ndim}.")
        if tuple(grad_out.shape) != expected_shape:
            raise ValueError(
                f"`grad_out` must have shape {expected_shape}, but got {tuple(grad_out.shape)}."
            )

    def _rms_norm(self, x: Tensor, gamma: Tensor) -> Tensor:
        """
        Apply RMSNorm over the last dimension and scale with gamma.

        Main feature:
            Computes x / sqrt(mean(x^2) + eps), then applies per-group gain.

        Inputs:
            x: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            gamma: float32 tensor of shape [hc_mult, d] on CPU

        Outputs:
            y: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
        """
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        gamma_broadcast = gamma.view(1, 1, gamma.shape[0], gamma.shape[1])
        return x_norm * gamma_broadcast

    def forward(self, h: Tensor, k: Tensor, gamma1: Tensor, gamma2: Tensor) -> Tensor:
        """
        Compute per-group dot product between RMS-normalized h and k.

        Main feature:
            Applies RMSNorm independently per [batch, token, hc_mult] group, then
            reduces over d using dot product.

        Inputs:
            h: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            k: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            gamma1: float32 tensor of shape [hc_mult, d] on CPU
            gamma2: float32 tensor of shape [hc_mult, d] on CPU

        Outputs:
            out: float32 tensor of shape [batch_size, seq_len, hc_mult] on CPU
        """
        self._validate_inputs(h, k, gamma1, gamma2)
        h_hat = self._rms_norm(h, gamma1)
        k_hat = self._rms_norm(k, gamma2)
        out = (h_hat * k_hat).sum(dim=-1)
        return out

    def manual_backward(
        self, h: Tensor, k: Tensor, gamma1: Tensor, gamma2: Tensor, grad_out: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute manual gradients for h, k, gamma1, gamma2 from upstream grad_out.

        Main feature:
            Implements closed-form backward equations for RMSNorm-dot on each
            [batch, token, hc_mult] group and accumulates gamma grads over [batch, token].

        Inputs:
            h: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            k: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            gamma1: float32 tensor of shape [hc_mult, d] on CPU
            gamma2: float32 tensor of shape [hc_mult, d] on CPU
            grad_out: float32 tensor of shape [batch_size, seq_len, hc_mult] on CPU

        Outputs:
            grad_h: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            grad_k: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            grad_gamma1: float32 tensor of shape [hc_mult, d] on CPU
            grad_gamma2: float32 tensor of shape [hc_mult, d] on CPU
        """
        self._validate_inputs(h, k, gamma1, gamma2)
        expected_grad_shape = (h.shape[0], h.shape[1], h.shape[2])
        self._validate_grad_out(grad_out, expected_grad_shape)

        d = h.shape[-1]

        h_rms = torch.sqrt(torch.mean(h * h, dim=-1, keepdim=True) + self.eps)
        k_rms = torch.sqrt(torch.mean(k * k, dim=-1, keepdim=True) + self.eps)
        h_hat = h / h_rms
        k_hat = k / k_rms

        gamma1_broadcast = gamma1[None, None, :, :]
        gamma2_broadcast = gamma2[None, None, :, :]
        u = h_hat * gamma1_broadcast
        v = k_hat * gamma2_broadcast
        y = (u * v).sum(dim=-1)

        grad_out_expanded = grad_out.unsqueeze(dim=-1)
        grad_gamma1 = (grad_out_expanded * (h_hat * v)).sum(dim=(0, 1))
        grad_gamma2 = (grad_out_expanded * (k_hat * u)).sum(dim=(0, 1))

        grad_h = (grad_out_expanded / h_rms) * (
            (gamma1_broadcast * v) - ((y.unsqueeze(dim=-1) / d) * h_hat)
        )
        grad_k = (grad_out_expanded / k_rms) * (
            (gamma2_broadcast * u) - ((y.unsqueeze(dim=-1) / d) * k_hat)
        )
        return grad_h, grad_k, grad_gamma1, grad_gamma2


def _reference_rmsnorm_dot(h: Tensor, k: Tensor, gamma1: Tensor, gamma2: Tensor, eps: float) -> Tensor:
    """
    Compute reference RMSNorm-dot output via a separate expression path.

    Main feature:
        Provides an independent correctness baseline for test assertions.

    Inputs:
        h: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
        k: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
        gamma1: float32 tensor of shape [hc_mult, d] on CPU
        gamma2: float32 tensor of shape [hc_mult, d] on CPU
        eps: float scalar epsilon for numerical stability

    Outputs:
        out_ref: float32 tensor of shape [batch_size, seq_len, hc_mult] on CPU
    """
    h_rms = torch.sqrt(torch.mean(torch.square(h), dim=-1, keepdim=True) + eps)
    k_rms = torch.sqrt(torch.mean(torch.square(k), dim=-1, keepdim=True) + eps)
    h_norm = (h / h_rms) * gamma1[None, None, :, :]
    k_norm = (k / k_rms) * gamma2[None, None, :, :]
    # Use the same reduction operator as forward() to avoid fp32 accumulation
    # order differences that can appear between einsum and sum.
    out_ref = (h_norm * k_norm).sum(dim=-1)
    return out_ref


def _relative_l2_error(actual: Tensor, expected: Tensor, denom_eps: float = 1e-12) -> float:
    """
    Compute relative L2 error between tensors of the same shape.

    Main feature:
        Returns ||actual - expected||_2 / (||expected||_2 + denom_eps).

    Inputs:
        actual: float32 tensor of shape [*] on CPU
        expected: float32 tensor of shape [*] on CPU
        denom_eps: float scalar added to denominator for zero-norm safety

    Outputs:
        rel_l2_error: float scalar
    """
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"`actual` and `expected` must share shape, but got {tuple(actual.shape)} "
            f"and {tuple(expected.shape)}."
        )
    numerator = torch.linalg.vector_norm((actual - expected).reshape(-1), ord=2)
    denominator = torch.linalg.vector_norm(expected.reshape(-1), ord=2)
    rel_l2_error = numerator / (denominator + denom_eps)
    return float(rel_l2_error.item())


def run_self_test() -> None:
    """
    Run a deterministic CPU/fp32 correctness test for RMSNormDotProduct.

    Main feature:
        Verifies output shape, dtype, device, numerical agreement, autograd
        backward, and manual-backward gradient matching.

    Inputs:
        None

    Outputs:
        None
    """
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 1024
    d = 128
    hc_mult = 4

    h = torch.randn(
        batch_size, seq_len, hc_mult, d, dtype=torch.float32, device="cpu", requires_grad=True
    )
    k = torch.randn(
        batch_size, seq_len, hc_mult, d, dtype=torch.float32, device="cpu", requires_grad=True
    )
    gamma1 = torch.randn(hc_mult, d, dtype=torch.float32, device="cpu", requires_grad=True)
    gamma2 = torch.randn(hc_mult, d, dtype=torch.float32, device="cpu", requires_grad=True)

    module = RMSNormDotProduct(eps=1e-5)
    out = module(h, k, gamma1, gamma2)

    expected_shape = (batch_size, seq_len, hc_mult)
    assert tuple(out.shape) == expected_shape, f"Expected shape {expected_shape}, got {tuple(out.shape)}."
    assert out.dtype == torch.float32, f"Expected dtype torch.float32, got {out.dtype}."
    assert out.device.type == "cpu", f"Expected CPU output, got {out.device}."

    out_ref = _reference_rmsnorm_dot(h, k, gamma1, gamma2, eps=1e-5)
    rel_l2_out_error = _relative_l2_error(out, out_ref)
    assert rel_l2_out_error <= 1e-6, f"Output mismatch vs reference, rel_l2={rel_l2_out_error:.6e}."

    loss = out.sum()
    loss.backward()
    assert h.grad is not None and tuple(h.grad.shape) == tuple(h.shape), "Missing or bad grad for h."
    assert k.grad is not None and tuple(k.grad.shape) == tuple(k.shape), "Missing or bad grad for k."
    assert gamma1.grad is not None and tuple(gamma1.grad.shape) == tuple(gamma1.shape), (
        "Missing or bad grad for gamma1."
    )
    assert gamma2.grad is not None and tuple(gamma2.grad.shape) == tuple(gamma2.shape), (
        "Missing or bad grad for gamma2."
    )

    grad_out = torch.ones_like(out)
    grad_h_manual, grad_k_manual, grad_gamma1_manual, grad_gamma2_manual = module.manual_backward(
        h.detach(), k.detach(), gamma1.detach(), gamma2.detach(), grad_out
    )
    rel_l2_h_grad_error = _relative_l2_error(h.grad, grad_h_manual)
    rel_l2_k_grad_error = _relative_l2_error(k.grad, grad_k_manual)
    rel_l2_gamma1_grad_error = _relative_l2_error(gamma1.grad, grad_gamma1_manual)
    rel_l2_gamma2_grad_error = _relative_l2_error(gamma2.grad, grad_gamma2_manual)

    assert rel_l2_h_grad_error <= 1e-6, (
        f"Manual backward mismatch for h, rel_l2={rel_l2_h_grad_error:.6e}."
    )
    assert rel_l2_k_grad_error <= 1e-6, (
        f"Manual backward mismatch for k, rel_l2={rel_l2_k_grad_error:.6e}."
    )
    assert rel_l2_gamma1_grad_error <= 1e-6, (
        f"Manual backward mismatch for gamma1, rel_l2={rel_l2_gamma1_grad_error:.6e}."
    )
    assert rel_l2_gamma2_grad_error <= 1e-6, (
        f"Manual backward mismatch for gamma2, rel_l2={rel_l2_gamma2_grad_error:.6e}."
    )

    max_abs_error = (out - out_ref).abs().max().item()
    max_abs_grad_h_error = (h.grad - grad_h_manual).abs().max().item()
    max_abs_grad_k_error = (k.grad - grad_k_manual).abs().max().item()
    max_abs_grad_gamma1_error = (gamma1.grad - grad_gamma1_manual).abs().max().item()
    max_abs_grad_gamma2_error = (gamma2.grad - grad_gamma2_manual).abs().max().item()
    grad_norm_h = h.grad.norm().item()
    grad_norm_k = k.grad.norm().item()
    grad_norm_gamma1 = gamma1.grad.norm().item()
    grad_norm_gamma2 = gamma2.grad.norm().item()
    print("Self-test passed")
    print(
        f"  summary: shape={tuple(out.shape)}, dtype={out.dtype}, "
        f"device={out.device}, loss={loss.item():.6e}"
    )
    print(f"  output error: rel_l2={rel_l2_out_error:.6e}, max_abs={max_abs_error:.6e}")
    print(
        f"  grad h error: rel_l2={rel_l2_h_grad_error:.6e}, "
        f"max_abs={max_abs_grad_h_error:.6e}, norm={grad_norm_h:.6e}"
    )
    print(
        f"  grad k error: rel_l2={rel_l2_k_grad_error:.6e}, "
        f"max_abs={max_abs_grad_k_error:.6e}, norm={grad_norm_k:.6e}"
    )
    print(
        f"  grad gamma1 error: rel_l2={rel_l2_gamma1_grad_error:.6e}, "
        f"max_abs={max_abs_grad_gamma1_error:.6e}, norm={grad_norm_gamma1:.6e}"
    )
    print(
        f"  grad gamma2 error: rel_l2={rel_l2_gamma2_grad_error:.6e}, "
        f"max_abs={max_abs_grad_gamma2_error:.6e}, norm={grad_norm_gamma2:.6e}"
    )


if __name__ == "__main__":
    run_self_test()
