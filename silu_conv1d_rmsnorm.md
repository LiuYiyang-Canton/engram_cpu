<!-- ==============================================================================
Author: Liu Yiyang
Date:   2026-02-10
============================================================================== -->

# SiLUConv1dRMSNorm 数学公式说明

## 前向传播

前向算子的信息如下：

- 算子输入：
  - $u \in \mathbb{R}^{B \times S \times H \times D}，数据类型为 fp32$
  - $\gamma \in \mathbb{R}^{H \times D}，数据类型为 fp32$
  - $W \in \mathbb{R}^{C \times 1 \times K}，数据类型为 fp32，其中 C = H \cdot D$
  - $\mathrm{actual\_seq\_len}$：Python list[list[int]]，外层长度为 $B$，第 $b$ 个元素为该 batch 行的累积长度
- 算子输出：
  - $y \in \mathbb{R}^{B \times S \times H \times D}，数据类型为 fp32$

其中 B 表示 batch 大小，S 表示序列长度，H 表示 HC_MULT，D 表示特征维度，K 表示卷积核大小，$\Delta$ 表示 dilation，$P=(K-1)\Delta$。

对每个 batch 行 $b$，定义边界序列：

$$
L_b=[l_{b,0},l_{b,1},\dots,l_{b,m_b}], \quad l_{b,0}=0,\quad l_{b,m_b}\le S
$$

第 $j$ 段长度为：

$$
T_{b,j}=l_{b,j+1}-l_{b,j}
$$

先进行 RMSNorm（按 D 维）：

$$
\mathrm{RMS}(u)_{bsh}=\sqrt{\frac{1}{D}\sum_{i=1}^{D}u_{bshi}^{2}+\varepsilon}
$$

$$
\hat{u}_{bshd}=\frac{u_{bshd}}{\mathrm{RMS}(u)_{bsh}}, \quad
x_{bshd}=\hat{u}_{bshd}\cdot \gamma_{hd}
$$

将 x 按通道展平为 $\tilde{x}\in\mathbb{R}^{B\times C\times S}$（$C=H\cdot D$）。
然后在每个 segment 内独立做 depthwise Conv1d（segment 间无信息泄漏）。

定义第 $(b,j)$ 段输入：

$$
\tilde{x}^{(b,j)}_{bc\tau}=\tilde{x}_{bc,\;l_{b,j}+\tau},
\quad \tau=0,1,\dots,T_{b,j}-1
$$

第 $(b,j)$ 段卷积（padding=P，越界按零）：

$$
z_{\mathrm{full},bc\tau}^{(b,j)}
=\sum_{k=0}^{K-1}
W_{c,0,k}\cdot \tilde{x}_{bc,\;\tau+k\Delta-P}^{(b,j)}
$$

对该段做因果裁剪（只保留前 $T_{b,j}$ 个位置）：

$$
z_{bc\tau}^{(b,j)}=z_{\mathrm{full},bc\tau}^{(b,j)},\quad \tau=0,1,\dots,T_{b,j}-1
$$

经过 SiLU：

$$
a_{bc\tau}^{(b,j)}=\mathrm{SiLU}(z_{bc\tau}^{(b,j)})
=z_{bc\tau}^{(b,j)}\cdot \sigma(z_{bc\tau}^{(b,j)})
$$

将各段散射回全局激活张量 $a\in\mathbb{R}^{B\times C\times S}$：

$$
a_{bc,\;l_{b,j}+\tau}=a_{bc\tau}^{(b,j)},\quad \tau=0,\dots,T_{b,j}-1
$$

对 padded tail（若 $l_{b,m_b}<S$）：

$$
a_{bct}=0,\quad t=l_{b,m_b},\dots,S-1
$$

$$
y_{bshd}=\mathrm{reshape}(a)_{bshd}+u_{bshd}
$$

因此 padded tail 上满足：

$$
y_{bthd}=u_{bthd},\quad t\in[l_{b,m_b},S)
$$

## 反向传播

反向算子的信息如下：

- 算子输入：
  - $u \in \mathbb{R}^{B \times S \times H \times D}，数据类型为 fp32$
  - $\gamma \in \mathbb{R}^{H \times D}，数据类型为 fp32$
  - $W \in \mathbb{R}^{C \times 1 \times K}，数据类型为 fp32$
  - $\mathrm{actual\_seq\_len}：与前向一致的边界列表
  - $\delta^y=\frac{\partial \mathcal{L}}{\partial y}\in\mathbb{R}^{B \times S \times H \times D}，数据类型为 fp32$
- 算子输出：
  - $\frac{\partial \mathcal{L}}{\partial u}\in\mathbb{R}^{B \times S \times H \times D}，数据类型为 fp32$
  - $\frac{\partial \mathcal{L}}{\partial \gamma}\in\mathbb{R}^{H \times D}，数据类型为 fp32$
  - $\frac{\partial \mathcal{L}}{\partial W}\in\mathbb{R}^{C \times 1 \times K}，数据类型为 fp32$
- 通过 recompute 得到的中间变量（反向前先重算）：
  - $\mathrm{RMS}(u), \hat{u}, x$
  - 各 segment 的 $z_{\mathrm{full}}^{(b,j)}, z^{(b,j)}$

先处理残差分支：

$$
\delta^{u,\mathrm{res}}=\delta^y
$$

将 $\delta^y$ 重排到 $\delta^a\in\mathbb{R}^{B\times C\times S}$。
对每个 segment $(b,j)$，在重算的 $z^{(b,j)}$ 上做 SiLU 反向：

$$
\delta z_{bc\tau}^{(b,j)}
=\delta a_{bc,\;l_{b,j}+\tau}\cdot
\sigma\!\left(z_{bc\tau}^{(b,j)}\right)\cdot
\left(1+z_{bc\tau}^{(b,j)}\cdot\left(1-\sigma\!\left(z_{bc\tau}^{(b,j)}\right)\right)\right)
$$

将每段裁剪前长度恢复为 $L^{\mathrm{full}}_{b,j}$（对应该段 conv1d 输出长度）：

$$
\delta z_{bc,0:T_{b,j}-1}^{\mathrm{full},(b,j)}=\delta z_{bc,0:T_{b,j}-1}^{(b,j)},\quad
\delta z_{bc,T_{b,j}:L^{\mathrm{full}}_{b,j}-1}^{\mathrm{full},(b,j)}=0
$$

每段卷积输入梯度（depthwise transposed convolution）：

$$
\delta x_{bc\tau}^{(b,j)}
=\sum_{k=0}^{K-1}
W_{c,0,k}\cdot
\delta z_{bc,\;\tau-k\Delta+P}^{\mathrm{full},(b,j)}
$$

再散射累加回全局 $\delta x\in\mathbb{R}^{B\times C\times S}$：

$$
\delta x_{bc,\;l_{b,j}+\tau}
\mathrel{+}= \delta x_{bc\tau}^{(b,j)},\quad \tau=0,\dots,T_{b,j}-1
$$

卷积权重梯度按段、按 kernel tap 累加（与实现一致）：

$$
\frac{\partial \mathcal{L}}{\partial W_{c,0,k}}
=\sum_{b}\sum_{j}\sum_{t=0}^{L^{\mathrm{full}}_{b,j}-1}
\delta z_{bct}^{\mathrm{full},(b,j)}
\cdot
\left(\tilde{x}^{\mathrm{pad},(b,j)}_{bc,\;t+k\Delta}\right)
$$

其中 $\tilde{x}^{\mathrm{pad},(b,j)}$ 为第 $(b,j)$ 段输入在时间维两侧各补 $P$ 个零。
对于 padded tail（$t\in[l_{b,m_b},S)$）没有 segment 覆盖，因此 conv 分支梯度贡献为 0。

把组装后的全局 $\delta^{x}$ 重排回 $\delta^{x}_{bshd}$ 后，进入 RMSNorm 反向（公式不变）：

$$
\delta^{\hat{u}}_{bshd}=\delta^{x}_{bshd}\cdot \gamma_{hd}
$$

$$
m_{bsh}=\frac{1}{D}\sum_{i=1}^{D}\delta^{\hat{u}}_{bshi}\hat{u}_{bshi}
$$

$$
\delta^{u,\mathrm{rms}}_{bshd}
=\frac{\delta^{\hat{u}}_{bshd}-m_{bsh}\hat{u}_{bshd}}
{\mathrm{RMS}(u)_{bsh}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \gamma_{hd}}
=\sum_{b,s}
\delta^{x}_{bshd}\hat{u}_{bshd}
$$

最终输入梯度为：

$$
\frac{\partial \mathcal{L}}{\partial u}
=\delta^{u,\mathrm{res}}+\delta^{u,\mathrm{rms}}
$$

## 实现约束说明

`actual_seq_len` 在实现中满足如下约束：

1. 必须是 Python `list`，长度为 `B`。
2. 每个元素必须是 Python `list[int]`。
3. 每个子列表必须以 `0` 开头。
4. 每个子列表必须严格递增（不允许相等）。
5. 每个值必须满足 `0 <= value <= S`。
6. 最后一个值允许 `< S`（表示 padded tail）。
7. `bool` 会被拒绝（即使 Python 中 `bool` 是 `int` 子类）。

## 与测试行为一致

当前实现与自测覆盖以下行为：

1. full-coverage packed case：各 batch 行最后一个边界等于 `S`。
2. padded-tail case：最后一个边界 `< S` 时，tail 上有 `y_{tail}=u_{tail}`。
3. 仅在 padded tail 注入上游梯度时，conv 分支对 $`\partial \mathcal{L}/\partial \gamma`$ 和
   $`\partial \mathcal{L}/\partial W`$ 贡献为 0。
4. 非法 `actual_seq_len`（长度不匹配、非递增、越界、类型错误等）会触发校验异常。
