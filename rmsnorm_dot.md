<!-- ==============================================================================
Author: yiyang
Date:   2026-02-10
============================================================================== -->

# RMSNormDotProduct 数学公式说明

## 前向传播

前向算子的信息如下：

- 算子输入：
  - $h \in \mathbb{R}^{B \times S \times H \times D}`，数据类型为 `fp32`$
  - $`k \in \mathbb{R}^{B \times S \times H \times D}`，数据类型为 `fp32`$
  - $`\gamma_1 \in \mathbb{R}^{H \times D}`，数据类型为 `fp32`$
  - $`\gamma_2 \in \mathbb{R}^{H \times D}`，数据类型为 `fp32`$
- 算子输出：
  - $`out \in \mathbb{R}^{B \times S \times H}`，数据类型为 `fp32`$

其中 `B` 表示 batch 大小，`S` 表示序列长度，`H` 表示 `HC_MULT`，`D` 表示特征维度。

对任意固定的 `(b, s, m)`（分别对应 batch、token、HC stream），定义向量：

$$
\mathbf{h}_{bsm} = h[b,s,m,:] \in \mathbb{R}^{D}, \quad
\mathbf{k}_{bsm} = k[b,s,m,:] \in \mathbb{R}^{D}
$$

$$
\mathbf{\gamma}_{1m} = \gamma_1[m,:] \in \mathbb{R}^{D}, \quad
\mathbf{\gamma}_{2m} = \gamma_2[m,:] \in \mathbb{R}^{D}
$$

先计算 RMS：

$$
\mathrm{RMS}(\mathbf{h}_{bsm})=\sqrt{\frac{1}{D}\sum_{i=1}^{D}h_{bsm,i}^{2}+\varepsilon}, \quad
\mathrm{RMS}(\mathbf{k}_{bsm})=\sqrt{\frac{1}{D}\sum_{i=1}^{D}k_{bsm,i}^{2}+\varepsilon}
$$

再计算归一化向量：

$$
\hat{\mathbf{h}}_{bsm}=\frac{\mathbf{h}_{bsm}}{\mathrm{RMS}(\mathbf{h}_{bsm})}, \quad
\hat{\mathbf{k}}_{bsm}=\frac{\mathbf{k}_{bsm}}{\mathrm{RMS}(\mathbf{k}_{bsm})}
$$

再做逐元素缩放：

$$
\mathbf{u}_{bsm}=\hat{\mathbf{h}}_{bsm}\odot\mathbf{\gamma}_{1m}, \quad
\mathbf{v}_{bsm}=\hat{\mathbf{k}}_{bsm}\odot\mathbf{\gamma}_{2m}
$$

最终输出为点积：

$$
y_{bsm}=\mathbf{u}_{bsm}^{\top}\mathbf{v}_{bsm}
=\sum_{i=1}^{D}u_{bsm,i}v_{bsm,i}
$$

因此 `out \in \mathbb{R}^{B \times S \times H}`，且 `out[b,s,m]=y_{bsm}`。

## 反向传播

反向算子的信息如下：

- 算子输入：
  - $`h \in \mathbb{R}^{B \times S \times H \times D}`，数据类型为 `fp32`$
  - $`k \in \mathbb{R}^{B \times S \times H \times D}`，数据类型为 `fp32`$
  - $`\gamma_1 \in \mathbb{R}^{H \times D}`，数据类型为 `fp32`$
  - $`\gamma_2 \in \mathbb{R}^{H \times D}`，数据类型为 `fp32`$
  - $`\delta \in \mathbb{R}^{B \times S \times H}`（即 `grad_out`），数据类型为 `fp32`$
- 算子输出：
  - $`\frac{\partial \mathcal{L}}{\partial h} \in \mathbb{R}^{B \times S \times H \times D}`，数据类型为 `fp32`$
  - $`\frac{\partial \mathcal{L}}{\partial k} \in \mathbb{R}^{B \times S \times H \times D}`，数据类型为 `fp32`$
  - $`\frac{\partial \mathcal{L}}{\partial \gamma_1} \in \mathbb{R}^{H \times D}`，数据类型为 `fp32`$
  - $`\frac{\partial \mathcal{L}}{\partial \gamma_2} \in \mathbb{R}^{H \times D}`，数据类型为 `fp32`$
- 通过 recompute 得到的中间变量（反向前先重算）：
  - $`\mathrm{RMS}(\mathbf{h}_{bsm}), \mathrm{RMS}(\mathbf{k}_{bsm})`$
  - $`\hat{\mathbf{h}}_{bsm}, \hat{\mathbf{k}}_{bsm}`$
  - $`\mathbf{u}_{bsm}, \mathbf{v}_{bsm}, y_{bsm}`$

设上游梯度为：

$$
\delta_{bsm}=\frac{\partial \mathcal{L}}{\partial y_{bsm}}
$$

对任意固定 `(b,s,m)`，有以下局部梯度公式。

先给出对 `\gamma` 的梯度（逐元素）：

$$
\frac{\partial y_{bsm}}{\partial \mathbf{\gamma}_{1m}}
= \hat{\mathbf{h}}_{bsm}\odot \mathbf{v}_{bsm}
$$

$$
\frac{\partial y_{bsm}}{\partial \mathbf{\gamma}_{2m}}
= \hat{\mathbf{k}}_{bsm}\odot \mathbf{u}_{bsm}
$$

乘以上游梯度并在 `b,s` 维度累加，得到参数梯度：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{\gamma}_{1m}}
=\sum_{b=1}^{B}\sum_{s=1}^{S}
\delta_{bsm}\left(\hat{\mathbf{h}}_{bsm}\odot \mathbf{v}_{bsm}\right)
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{\gamma}_{2m}}
=\sum_{b=1}^{B}\sum_{s=1}^{S}
\delta_{bsm}\left(\hat{\mathbf{k}}_{bsm}\odot \mathbf{u}_{bsm}\right)
$$

对输入向量的梯度为：

$$
\frac{\partial y_{bsm}}{\partial \mathbf{h}_{bsm}}
=\frac{1}{\mathrm{RMS}(\mathbf{h}_{bsm})}
\left[
\left(\mathbf{\gamma}_{1m}\odot \mathbf{v}_{bsm}\right)
-\frac{y_{bsm}}{D}\hat{\mathbf{h}}_{bsm}
\right]
$$

$$
\frac{\partial y_{bsm}}{\partial \mathbf{k}_{bsm}}
=\frac{1}{\mathrm{RMS}(\mathbf{k}_{bsm})}
\left[
\left(\mathbf{\gamma}_{2m}\odot \mathbf{u}_{bsm}\right)
-\frac{y_{bsm}}{D}\hat{\mathbf{k}}_{bsm}
\right]
$$

乘以上游梯度后得到损失对输入的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{bsm}}
=\delta_{bsm}\cdot
\frac{\partial y_{bsm}}{\partial \mathbf{h}_{bsm}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{k}_{bsm}}
=\delta_{bsm}\cdot
\frac{\partial y_{bsm}}{\partial \mathbf{k}_{bsm}}
$$

以上公式按每个 batch、每个 token、每个 HC stream 独立成立。
