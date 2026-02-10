# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-02-10
# Purpose:
# ==============================================================================

"""SiLU(Conv1D(RMSNorm(u))) + u implemented on CPU float32 tensors."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch import nn


class SiLUConv1dRMSNorm(nn.Module):
    """Compute y = SiLU(Conv1D(RMSNorm(u))) + u with causal depthwise Conv1d."""

    def __init__(self, kernel_size: int, dilation: int, eps: float = 1e-6) -> None:
        """
        Initialize the SiLU-Conv1d-RMSNorm module configuration.

        Main feature:
            Stores convolution and RMSNorm hyperparameters for forward compute.

        Inputs:
            kernel_size: int scalar kernel size for depthwise Conv1d, must be > 0
            dilation: int scalar dilation for depthwise Conv1d, must be > 0
            eps: float scalar epsilon added in RMS denominator, must be >= 0

        Outputs:
            None
        """
        super().__init__()
        if not isinstance(kernel_size, int) or isinstance(kernel_size, bool):
            raise TypeError(f"`kernel_size` must be int, but got type {type(kernel_size)}.")
        if not isinstance(dilation, int) or isinstance(dilation, bool):
            raise TypeError(f"`dilation` must be int, but got type {type(dilation)}.")
        if kernel_size <= 0:
            raise ValueError(f"`kernel_size` must be > 0, but got {kernel_size}.")
        if dilation <= 0:
            raise ValueError(f"`dilation` must be > 0, but got {dilation}.")
        if eps < 0:
            raise ValueError(f"`eps` must be >= 0, but got {eps}.")

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.eps = float(eps)

    def _validate_inputs(self, u: Tensor, gamma: Tensor, conv_weight: Tensor) -> None:
        """
        Validate tensor device, dtype, and shape constraints for forward pass.

        Main feature:
            Enforces CPU + float32 tensors and exact shape compatibility.

        Inputs:
            u: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            gamma: float32 tensor of shape [hc_mult, d] on CPU
            conv_weight: float32 tensor of shape [hc_mult * d, 1, kernel_size] on CPU

        Outputs:
            None
        """
        tensors = {"u": u, "gamma": gamma, "conv_weight": conv_weight}
        for name, tensor in tensors.items():
            if tensor.device.type != "cpu":
                raise ValueError(f"`{name}` must be on CPU, but got device {tensor.device}.")
            if tensor.dtype != torch.float32:
                raise TypeError(f"`{name}` must be torch.float32, but got dtype {tensor.dtype}.")

        if u.ndim != 4:
            raise ValueError(f"`u` must be rank-4, but got ndim={u.ndim}.")
        if gamma.ndim != 2:
            raise ValueError(f"`gamma` must be rank-2, but got ndim={gamma.ndim}.")
        if conv_weight.ndim != 3:
            raise ValueError(f"`conv_weight` must be rank-3, but got ndim={conv_weight.ndim}.")

        batch_size, seq_len, hc_mult, d = u.shape
        if batch_size <= 0 or seq_len <= 0 or hc_mult <= 0 or d <= 0:
            raise ValueError(f"All dimensions in `u` must be positive, but got shape {tuple(u.shape)}.")

        expected_gamma_shape = (hc_mult, d)
        if tuple(gamma.shape) != expected_gamma_shape:
            raise ValueError(
                f"`gamma` must have shape {expected_gamma_shape}, but got {tuple(gamma.shape)}."
            )

        total_channels = hc_mult * d
        expected_conv_weight_shape = (total_channels, 1, self.kernel_size)
        if tuple(conv_weight.shape) != expected_conv_weight_shape:
            raise ValueError(
                f"`conv_weight` must have shape {expected_conv_weight_shape}, "
                f"but got {tuple(conv_weight.shape)}."
            )

    @staticmethod
    def _validate_grad_out(grad_out: Tensor, expected_shape: tuple[int, int, int, int]) -> None:
        """
        Validate upstream gradient tensor for manual backward.

        Main feature:
            Enforces CPU + float32 and exact [batch_size, seq_len, hc_mult, d] shape.

        Inputs:
            grad_out: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            expected_shape: tuple[int, int, int, int] for [batch_size, seq_len, hc_mult, d]

        Outputs:
            None
        """
        if grad_out.device.type != "cpu":
            raise ValueError(f"`grad_out` must be on CPU, but got device {grad_out.device}.")
        if grad_out.dtype != torch.float32:
            raise TypeError(f"`grad_out` must be torch.float32, but got dtype {grad_out.dtype}.")
        if grad_out.ndim != 4:
            raise ValueError(f"`grad_out` must be rank-4, but got ndim={grad_out.ndim}.")
        if tuple(grad_out.shape) != expected_shape:
            raise ValueError(
                f"`grad_out` must have shape {expected_shape}, but got {tuple(grad_out.shape)}."
            )

    @staticmethod
    def _validate_actual_seq_len(
        actual_seq_len: list[list[int]], batch_size: int, seq_len: int
    ) -> None:
        """
        Validate packed sub-sequence cumulative boundaries for each batch row.

        Main feature:
            Enforces Python-list cumsum layout with strict boundaries and optional padded tail.

        Inputs:
            actual_seq_len: list of length [batch_size], each item is list[int] cumsums
                for one batch row; each row must start with 0 and be strictly increasing,
                and all values must be in [0, seq_len]
            batch_size: int scalar number of batch rows
            seq_len: int scalar maximum sequence length per batch row

        Outputs:
            None
        """
        if not isinstance(actual_seq_len, list):
            raise TypeError(
                f"`actual_seq_len` must be a Python list, but got type {type(actual_seq_len)}."
            )
        if len(actual_seq_len) != batch_size:
            raise ValueError(
                f"`actual_seq_len` must have length {batch_size}, "
                f"but got {len(actual_seq_len)}."
            )

        for batch_index, row in enumerate(actual_seq_len):
            if not isinstance(row, list):
                raise TypeError(
                    f"`actual_seq_len[{batch_index}]` must be a Python list, "
                    f"but got type {type(row)}."
                )
            if len(row) == 0:
                raise ValueError(
                    f"`actual_seq_len[{batch_index}]` must not be empty."
                )
            if row[0] != 0:
                raise ValueError(
                    f"`actual_seq_len[{batch_index}]` must start with 0, "
                    f"but got {row[0]}."
                )

            for value_index, value in enumerate(row):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(
                        f"`actual_seq_len[{batch_index}][{value_index}]` must be int, "
                        f"but got type {type(value)}."
                    )
                if value < 0 or value > seq_len:
                    raise ValueError(
                        f"`actual_seq_len[{batch_index}][{value_index}]` must be within "
                        f"[0, {seq_len}], but got {value}."
                    )

            for left, right in zip(row, row[1:]):
                if right <= left:
                    raise ValueError(
                        f"`actual_seq_len[{batch_index}]` must be strictly increasing, "
                        f"but found {left} then {right}."
                    )

    def _rms_norm(self, u: Tensor, gamma: Tensor) -> Tensor:
        """
        Apply RMSNorm over the last dimension and scale by gamma.

        Main feature:
            Computes RMSNorm per [batch, token, hc_mult] group over d.

        Inputs:
            u: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            gamma: float32 tensor of shape [hc_mult, d] on CPU

        Outputs:
            u_norm: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
        """
        rms = torch.sqrt(torch.mean(u * u, dim=-1, keepdim=True) + self.eps)
        u_hat = u / rms
        u_norm = u_hat * gamma[None, None, :, :]
        return u_norm

    def forward(
        self, u: Tensor, gamma: Tensor, conv_weight: Tensor, actual_seq_len: list[list[int]]
    ) -> Tensor:
        """
        Compute y = SiLU(Conv1D(RMSNorm(u))) + u with causal depthwise convolution.

        Main feature:
            Applies RMSNorm, then runs depthwise Conv1d+SiLU independently in each
            packed sub-sequence, and finally adds residual input u.

        Inputs:
            u: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            gamma: float32 tensor of shape [hc_mult, d] on CPU
            conv_weight: float32 tensor of shape [hc_mult * d, 1, kernel_size] on CPU
            actual_seq_len: list of length [batch_size], each item is list[int]
                cumulative sub-sequence boundaries for one batch row, starting at 0

        Outputs:
            y: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
        """
        self._validate_inputs(u, gamma, conv_weight)
        batch_size, seq_len, hc_mult, d = u.shape
        self._validate_actual_seq_len(actual_seq_len, batch_size=batch_size, seq_len=seq_len)

        total_channels = hc_mult * d
        conv_pad = (self.kernel_size - 1) * self.dilation

        u_norm = self._rms_norm(u, gamma)
        x = u_norm.reshape(batch_size, seq_len, total_channels).transpose(1, 2).contiguous()

        act_out = torch.zeros_like(x)
        for batch_index, row_cumsums in enumerate(actual_seq_len):
            for segment_index in range(len(row_cumsums) - 1):
                segment_start = row_cumsums[segment_index]
                segment_end = row_cumsums[segment_index + 1]
                segment_len = segment_end - segment_start
                x_segment = x[batch_index : batch_index + 1, :, segment_start:segment_end]
                conv_segment = functional.conv1d(
                    input=x_segment,
                    weight=conv_weight,
                    bias=None,
                    stride=1,
                    padding=conv_pad,
                    dilation=self.dilation,
                    groups=total_channels,
                )
                conv_segment = conv_segment[..., :segment_len]
                act_out[batch_index : batch_index + 1, :, segment_start:segment_end] = functional.silu(
                    conv_segment
                )

        act_out_bshd = act_out.transpose(1, 2).reshape(batch_size, seq_len, hc_mult, d).contiguous()
        y = act_out_bshd + u
        return y

    def manual_backward(
        self,
        u: Tensor,
        gamma: Tensor,
        conv_weight: Tensor,
        actual_seq_len: list[list[int]],
        grad_out: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute manual backward gradients for u, gamma, and conv_weight.

        Main feature:
            Recomputes forward intermediates and applies explicit chain-rule gradients
            through residual, SiLU, packed depthwise Conv1d, and RMSNorm.

        Inputs:
            u: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            gamma: float32 tensor of shape [hc_mult, d] on CPU
            conv_weight: float32 tensor of shape [hc_mult * d, 1, kernel_size] on CPU
            actual_seq_len: list of length [batch_size], each item is list[int]
                cumulative sub-sequence boundaries for one batch row, starting at 0
            grad_out: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU

        Outputs:
            grad_u: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
            grad_gamma: float32 tensor of shape [hc_mult, d] on CPU
            grad_conv_weight: float32 tensor of shape [hc_mult * d, 1, kernel_size] on CPU
        """
        self._validate_inputs(u, gamma, conv_weight)
        batch_size, seq_len, hc_mult, d = u.shape
        self._validate_actual_seq_len(actual_seq_len, batch_size=batch_size, seq_len=seq_len)
        self._validate_grad_out(grad_out, expected_shape=tuple(u.shape))

        total_channels = hc_mult * d
        conv_pad = (self.kernel_size - 1) * self.dilation

        # Recompute forward intermediates.
        rms = torch.sqrt(torch.mean(u * u, dim=-1, keepdim=True) + self.eps)
        u_hat = u / rms
        u_norm = u_hat * gamma[None, None, :, :]
        x = u_norm.reshape(batch_size, seq_len, total_channels).transpose(1, 2).contiguous()

        # Residual branch gradient.
        grad_u_residual = grad_out

        # SiLU + packed Conv1d branch gradient.
        grad_act = grad_out.reshape(batch_size, seq_len, total_channels).transpose(1, 2).contiguous()
        grad_x = torch.zeros_like(x)
        grad_conv_weight = torch.zeros_like(conv_weight)
        for batch_index, row_cumsums in enumerate(actual_seq_len):
            for segment_index in range(len(row_cumsums) - 1):
                segment_start = row_cumsums[segment_index]
                segment_end = row_cumsums[segment_index + 1]
                segment_len = segment_end - segment_start
                x_segment = x[batch_index : batch_index + 1, :, segment_start:segment_end]
                z_full_segment = functional.conv1d(
                    input=x_segment,
                    weight=conv_weight,
                    bias=None,
                    stride=1,
                    padding=conv_pad,
                    dilation=self.dilation,
                    groups=total_channels,
                )
                z_segment = z_full_segment[..., :segment_len]

                grad_act_segment = grad_act[batch_index : batch_index + 1, :, segment_start:segment_end]
                sigmoid_z = torch.sigmoid(z_segment)
                silu_prime = sigmoid_z * (1.0 + z_segment * (1.0 - sigmoid_z))
                grad_z_segment = grad_act_segment * silu_prime

                grad_z_full_segment = torch.zeros_like(z_full_segment)
                grad_z_full_segment[..., :segment_len] = grad_z_segment

                grad_x_segment = functional.conv_transpose1d(
                    input=grad_z_full_segment,
                    weight=conv_weight,
                    bias=None,
                    stride=1,
                    padding=conv_pad,
                    dilation=self.dilation,
                    groups=total_channels,
                )
                grad_x[batch_index : batch_index + 1, :, segment_start:segment_end] += grad_x_segment

                full_segment_len = z_full_segment.shape[-1]
                x_segment_pad = functional.pad(x_segment, (conv_pad, conv_pad))
                for kernel_index in range(self.kernel_size):
                    begin = kernel_index * self.dilation
                    end = begin + full_segment_len
                    x_slice = x_segment_pad[..., begin:end]
                    grad_conv_weight[:, 0, kernel_index] += (grad_z_full_segment * x_slice).sum(
                        dim=(0, 2)
                    )

        # Backprop from conv input to RMSNorm output layout.
        grad_u_norm = (
            grad_x.transpose(1, 2).reshape(batch_size, seq_len, hc_mult, d).contiguous()
        )

        # RMSNorm backward.
        grad_u_hat = grad_u_norm * gamma[None, None, :, :]
        projection_mean = torch.mean(grad_u_hat * u_hat, dim=-1, keepdim=True)
        grad_u_rms = (grad_u_hat - projection_mean * u_hat) / rms
        grad_gamma = (grad_u_norm * u_hat).sum(dim=(0, 1))

        grad_u = grad_u_residual + grad_u_rms
        return grad_u, grad_gamma, grad_conv_weight


def _reference_forward(
    u: Tensor,
    gamma: Tensor,
    conv_weight: Tensor,
    actual_seq_len: list[list[int]],
    kernel_size: int,
    dilation: int,
    eps: float,
) -> Tensor:
    """
    Compute an independent forward reference using nn.Conv1d depthwise module.

    Main feature:
        Rebuilds forward math in a separate path for numerical verification.

    Inputs:
        u: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
        gamma: float32 tensor of shape [hc_mult, d] on CPU
        conv_weight: float32 tensor of shape [hc_mult * d, 1, kernel_size] on CPU
        actual_seq_len: list of length [batch_size], each item is list[int]
            cumulative sub-sequence boundaries for one batch row, starting at 0
        kernel_size: int scalar kernel size
        dilation: int scalar dilation
        eps: float scalar epsilon for RMSNorm

    Outputs:
        out_ref: float32 tensor of shape [batch_size, seq_len, hc_mult, d] on CPU
    """
    batch_size, seq_len, hc_mult, d = u.shape
    SiLUConv1dRMSNorm._validate_actual_seq_len(
        actual_seq_len, batch_size=batch_size, seq_len=seq_len
    )
    total_channels = hc_mult * d

    rms = torch.sqrt(torch.mean(torch.square(u), dim=-1, keepdim=True) + eps)
    u_norm = (u / rms) * gamma[None, None, :, :]
    x = u_norm.reshape(batch_size, seq_len, total_channels).transpose(1, 2).contiguous()

    ref_conv = nn.Conv1d(
        in_channels=total_channels,
        out_channels=total_channels,
        kernel_size=kernel_size,
        groups=total_channels,
        bias=False,
        padding=(kernel_size - 1) * dilation,
        dilation=dilation,
    )
    with torch.no_grad():
        ref_conv.weight.copy_(conv_weight)

    act_ref = torch.zeros_like(x)
    for batch_index, row_cumsums in enumerate(actual_seq_len):
        for segment_index in range(len(row_cumsums) - 1):
            segment_start = row_cumsums[segment_index]
            segment_end = row_cumsums[segment_index + 1]
            segment_len = segment_end - segment_start
            x_segment = x[batch_index : batch_index + 1, :, segment_start:segment_end]
            conv_ref_segment = ref_conv(x_segment)
            conv_ref_segment = conv_ref_segment[..., :segment_len]
            act_ref[batch_index : batch_index + 1, :, segment_start:segment_end] = functional.silu(
                conv_ref_segment
            )

    out_ref = act_ref.transpose(1, 2).reshape(batch_size, seq_len, hc_mult, d).contiguous() + u
    return out_ref


def _assert_raises(
    callback: Callable[[], None], expected_exception: type[BaseException] | tuple[type[BaseException], ...]
) -> None:
    """
    Assert that callback raises one of the expected exception types.

    Main feature:
        Executes callback and fails if expected exception is not raised.

    Inputs:
        callback: callable object of shape [] -> None that triggers validation logic
        expected_exception: exception type or tuple of exception types

    Outputs:
        None
    """
    try:
        callback()
    except expected_exception:
        return
    except Exception as error:
        raise AssertionError(
            f"Expected {expected_exception}, but got {type(error)}: {error}"
        ) from error
    raise AssertionError(f"Expected {expected_exception}, but no exception was raised.")


def _run_single_self_test_case(case_name: str, actual_seq_len: list[list[int]]) -> None:
    """
    Run forward and backward correctness checks for one packed-sequence test case.

    Main feature:
        Compares module forward/manual backward against independent reference/autograd.

    Inputs:
        case_name: str scalar test case identifier
        actual_seq_len: list of length [batch_size], each item is list[int]
            cumulative sub-sequence boundaries for one batch row, starting at 0

    Outputs:
        None
    """
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 1024
    hc_mult = 4
    d = 128
    kernel_size = 4
    dilation = 2
    total_channels = hc_mult * d

    u = torch.randn(
        batch_size, seq_len, hc_mult, d, dtype=torch.float32, device="cpu", requires_grad=True
    )
    gamma = torch.randn(hc_mult, d, dtype=torch.float32, device="cpu", requires_grad=True)
    conv_weight = torch.randn(
        total_channels, 1, kernel_size, dtype=torch.float32, device="cpu", requires_grad=True
    )

    module = SiLUConv1dRMSNorm(kernel_size=kernel_size, dilation=dilation, eps=1e-6)
    out = module(u, gamma, conv_weight, actual_seq_len)

    expected_shape = (batch_size, seq_len, hc_mult, d)
    assert tuple(out.shape) == expected_shape, f"Expected shape {expected_shape}, got {tuple(out.shape)}."
    assert out.dtype == torch.float32, f"Expected dtype torch.float32, got {out.dtype}."
    assert out.device.type == "cpu", f"Expected CPU output, got {out.device}."

    out_ref = _reference_forward(
        u=u.detach(),
        gamma=gamma.detach(),
        conv_weight=conv_weight.detach(),
        actual_seq_len=actual_seq_len,
        kernel_size=kernel_size,
        dilation=dilation,
        eps=1e-6,
    )
    rel_l2_out_error = _relative_l2_error(out.detach(), out_ref)
    max_abs_out_error = (out.detach() - out_ref).abs().max().item()
    assert rel_l2_out_error <= 1e-6, (
        f"[{case_name}] Forward mismatch, rel_l2={rel_l2_out_error:.6e}."
    )

    loss = out.sum()
    loss.backward()
    assert u.grad is not None and tuple(u.grad.shape) == tuple(u.shape), "Missing or bad grad for u."
    assert gamma.grad is not None and tuple(gamma.grad.shape) == tuple(gamma.shape), (
        "Missing or bad grad for gamma."
    )
    assert conv_weight.grad is not None and tuple(conv_weight.grad.shape) == tuple(conv_weight.shape), (
        "Missing or bad grad for conv_weight."
    )

    grad_u_autograd = u.grad.detach().clone()
    grad_gamma_autograd = gamma.grad.detach().clone()
    grad_conv_weight_autograd = conv_weight.grad.detach().clone()

    grad_out = torch.ones_like(out)
    grad_u_manual, grad_gamma_manual, grad_conv_weight_manual = module.manual_backward(
        u=u.detach(),
        gamma=gamma.detach(),
        conv_weight=conv_weight.detach(),
        actual_seq_len=actual_seq_len,
        grad_out=grad_out,
    )

    rel_l2_grad_u_error = _relative_l2_error(grad_u_manual, grad_u_autograd)
    rel_l2_grad_gamma_error = _relative_l2_error(grad_gamma_manual, grad_gamma_autograd)
    rel_l2_grad_conv_weight_error = _relative_l2_error(
        grad_conv_weight_manual, grad_conv_weight_autograd
    )
    max_abs_grad_u_error = (grad_u_manual - grad_u_autograd).abs().max().item()
    max_abs_grad_gamma_error = (grad_gamma_manual - grad_gamma_autograd).abs().max().item()
    max_abs_grad_conv_weight_error = (
        grad_conv_weight_manual - grad_conv_weight_autograd
    ).abs().max().item()

    assert rel_l2_grad_u_error <= 1e-6, (
        f"[{case_name}] Manual backward mismatch for u, rel_l2={rel_l2_grad_u_error:.6e}."
    )
    assert rel_l2_grad_gamma_error <= 1e-6, (
        f"[{case_name}] Manual backward mismatch for gamma, rel_l2={rel_l2_grad_gamma_error:.6e}."
    )
    assert rel_l2_grad_conv_weight_error <= 1e-6, (
        f"[{case_name}] Manual backward mismatch for conv_weight, "
        f"rel_l2={rel_l2_grad_conv_weight_error:.6e}."
    )

    print(f"Case {case_name} passed")
    print(f"  output error: rel_l2={rel_l2_out_error:.6e}, max_abs={max_abs_out_error:.6e}")
    print(f"  grad u error: rel_l2={rel_l2_grad_u_error:.6e}, max_abs={max_abs_grad_u_error:.6e}")
    print(
        "  grad gamma error: "
        f"rel_l2={rel_l2_grad_gamma_error:.6e}, max_abs={max_abs_grad_gamma_error:.6e}"
    )
    print(
        "  grad conv_weight error: "
        f"rel_l2={rel_l2_grad_conv_weight_error:.6e}, "
        f"max_abs={max_abs_grad_conv_weight_error:.6e}"
    )


def _run_padded_tail_behavior_test(actual_seq_len: list[list[int]]) -> None:
    """
    Validate padded-tail behavior for forward identity and conv-path gradients.

    Main feature:
        Checks that padded tails bypass conv branch and only keep residual gradient.

    Inputs:
        actual_seq_len: list of length [batch_size], each item is list[int]
            cumulative sub-sequence boundaries for one batch row, starting at 0

    Outputs:
        None
    """
    torch.manual_seed(1)

    batch_size = 2
    seq_len = 1024
    hc_mult = 4
    d = 128
    kernel_size = 4
    dilation = 2
    total_channels = hc_mult * d

    u = torch.randn(batch_size, seq_len, hc_mult, d, dtype=torch.float32, device="cpu")
    gamma = torch.randn(hc_mult, d, dtype=torch.float32, device="cpu")
    conv_weight = torch.randn(total_channels, 1, kernel_size, dtype=torch.float32, device="cpu")

    module = SiLUConv1dRMSNorm(kernel_size=kernel_size, dilation=dilation, eps=1e-6)
    out = module(u, gamma, conv_weight, actual_seq_len)

    tail_grad_out = torch.zeros_like(out)
    has_padded_tail = False
    for batch_index, row_cumsums in enumerate(actual_seq_len):
        tail_start = row_cumsums[-1]
        if tail_start < seq_len:
            has_padded_tail = True
            tail_delta = (out[batch_index, tail_start:, :, :] - u[batch_index, tail_start:, :, :]).abs()
            assert tail_delta.max().item() <= 1e-7, (
                f"Padded tail output mismatch for batch {batch_index}, "
                f"max_abs={tail_delta.max().item():.6e}."
            )
            tail_grad_out[batch_index, tail_start:, :, :] = 1.0

    assert has_padded_tail, "Expected at least one padded tail in padded-tail test case."

    grad_u_manual, grad_gamma_manual, grad_conv_weight_manual = module.manual_backward(
        u=u,
        gamma=gamma,
        conv_weight=conv_weight,
        actual_seq_len=actual_seq_len,
        grad_out=tail_grad_out,
    )

    u_auto = u.detach().clone().requires_grad_(True)
    gamma_auto = gamma.detach().clone().requires_grad_(True)
    conv_weight_auto = conv_weight.detach().clone().requires_grad_(True)
    out_auto = module(u_auto, gamma_auto, conv_weight_auto, actual_seq_len)
    loss_tail = (out_auto * tail_grad_out).sum()
    loss_tail.backward()

    assert u_auto.grad is not None, "Missing autograd gradient for u in padded-tail test."
    assert gamma_auto.grad is not None, "Missing autograd gradient for gamma in padded-tail test."
    assert conv_weight_auto.grad is not None, "Missing autograd gradient for conv_weight in padded-tail test."

    grad_u_auto = u_auto.grad.detach()
    grad_gamma_auto = gamma_auto.grad.detach()
    grad_conv_weight_auto = conv_weight_auto.grad.detach()

    assert (grad_u_manual - grad_u_auto).abs().max().item() <= 1e-6, (
        "Padded-tail manual/autograd mismatch for grad_u, "
        f"max_abs={(grad_u_manual - grad_u_auto).abs().max().item():.6e}."
    )
    assert (grad_gamma_manual - grad_gamma_auto).abs().max().item() <= 1e-6, (
        "Padded-tail manual/autograd mismatch for grad_gamma, "
        f"max_abs={(grad_gamma_manual - grad_gamma_auto).abs().max().item():.6e}."
    )
    assert (grad_conv_weight_manual - grad_conv_weight_auto).abs().max().item() <= 1e-6, (
        "Padded-tail manual/autograd mismatch for grad_conv_weight, "
        f"max_abs={(grad_conv_weight_manual - grad_conv_weight_auto).abs().max().item():.6e}."
    )

    assert grad_gamma_manual.abs().max().item() <= 1e-7, (
        f"Padded-tail grad_gamma should be zero, max_abs={grad_gamma_manual.abs().max().item():.6e}."
    )
    assert grad_conv_weight_manual.abs().max().item() <= 1e-7, (
        "Padded-tail grad_conv_weight should be zero, "
        f"max_abs={grad_conv_weight_manual.abs().max().item():.6e}."
    )
    assert (grad_u_manual - tail_grad_out).abs().max().item() <= 1e-6, (
        f"Padded-tail grad_u should match residual grad_out, "
        f"max_abs={(grad_u_manual - tail_grad_out).abs().max().item():.6e}."
    )

    print("Case padded_tail_behavior passed")


def _run_actual_seq_len_validation_test() -> None:
    """
    Run input validation checks for actual_seq_len contract enforcement.

    Main feature:
        Verifies expected failures for malformed packed sub-sequence boundaries.

    Inputs:
        None

    Outputs:
        None
    """
    batch_size = 2
    seq_len = 16
    hc_mult = 2
    d = 8
    kernel_size = 3
    dilation = 1
    total_channels = hc_mult * d

    u = torch.randn(batch_size, seq_len, hc_mult, d, dtype=torch.float32, device="cpu")
    gamma = torch.randn(hc_mult, d, dtype=torch.float32, device="cpu")
    conv_weight = torch.randn(total_channels, 1, kernel_size, dtype=torch.float32, device="cpu")
    module = SiLUConv1dRMSNorm(kernel_size=kernel_size, dilation=dilation, eps=1e-6)

    _assert_raises(
        lambda: module(u, gamma, conv_weight, [[0, seq_len]]),
        ValueError,
    )
    _assert_raises(
        lambda: module(u, gamma, conv_weight, [[1, seq_len], [0, seq_len]]),
        ValueError,
    )
    _assert_raises(
        lambda: module(u, gamma, conv_weight, [[0, 4, 4], [0, seq_len]]),
        ValueError,
    )
    _assert_raises(
        lambda: module(u, gamma, conv_weight, [[0, seq_len + 1], [0, seq_len]]),
        ValueError,
    )
    _assert_raises(
        lambda: module(u, gamma, conv_weight, [[0, 4.0, seq_len], [0, seq_len]]),
        TypeError,
    )
    _assert_raises(
        lambda: module(u, gamma, conv_weight, [[0, True, seq_len], [0, seq_len]]),
        TypeError,
    )
    _assert_raises(
        lambda: module(u, gamma, conv_weight, [[0, seq_len], (0, seq_len)]),  # type: ignore[list-item]
        TypeError,
    )

    print("Case actual_seq_len_validation passed")


def _relative_l2_error(actual: Tensor, expected: Tensor, denom_eps: float = 1e-12) -> float:
    """
    Compute relative L2 error between tensors with identical shape.

    Main feature:
        Returns ||actual - expected||_2 / (||expected||_2 + denom_eps).

    Inputs:
        actual: float32 tensor of shape [*] on CPU
        expected: float32 tensor of shape [*] on CPU
        denom_eps: float scalar for denominator stability

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
    Run deterministic forward and autograd smoke tests for SiLUConv1dRMSNorm.

    Main feature:
        Validates output properties, reference agreement, and manual backward
        gradients against autograd via relative L2 error.

    Inputs:
        None

    Outputs:
        None
    """
    _run_single_self_test_case(
        case_name="full_coverage",
        actual_seq_len=[[0, 128, 512, 1024], [0, 256, 768, 1024]],
    )
    _run_single_self_test_case(
        case_name="padded_tail",
        actual_seq_len=[[0, 300, 700], [0, 256, 900]],
    )
    _run_padded_tail_behavior_test(actual_seq_len=[[0, 300, 700], [0, 256, 900]])
    _run_actual_seq_len_validation_test()
    print("Self-test passed")


if __name__ == "__main__":
    run_self_test()
