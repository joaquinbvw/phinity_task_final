# Artificial Neuron MAC (Fixed-Point, Serial) — Specification

## 1. Overview

Artificial neurons are basic computational units of modern neural networks. Conceptually, each neuron receives a set of input values, multiplies each input by an associated weight, adds all these contributions together, and then adds a bias term. The result of this weighted sum is often passed through an activation function, which introduces nonlinearity and helps the network approximate complex relationships.

In a typical setting, the inputs represent features of a data instance. Each weight encodes how strongly a particular input contributes to the neuron's decision: a positive weight reinforces the input, a negative weight suppresses it, and a weight near zero means the input has little influence. The bias term acts as a baseline offset, shifting the neuron's decision threshold independently of the inputs.

Several concepts are central to this behavior:

- **Bias**: a constant term added to the weighted sum of inputs. It allows the neuron to activate even when all inputs are zero and shifts the decision threshold.
- **Activation function**: a (usually nonlinear) function applied to the weighted sum plus bias. Common choices include ReLU, leaky ReLU, hard-tanh–style clamping, and the identity function (no nonlinearity).
- **Saturation / clipping**: mechanisms that limit the output to lie within a representable range, preventing unbounded growth.
- **Sparsity / masking**: the possibility of treating some inputs as "inactive" (multiplying them by zero via a mask). This reflects common practices such as pruning connections, enforcing structured sparsity, or gating subsets of features.
- **Serial / sequential computation**: hardware may trade throughput for smaller area by doing operations over multiple cycles (e.g., accumulating one product per element, and/or using a multi-cycle multiplier). This changes timing/latency but not the mathematical result.

A widely used activation in modern deep learning is the **Rectified Linear Unit (ReLU)**. ReLU sets all negative inputs to zero and passes positive inputs unchanged. Variants such as **leaky ReLU** keep a small non-zero slope for negative values, and **hard-tanh** clamps values to a bounded interval (often approximating a smooth tanh with piecewise-linear segments).

While the conceptual description of neurons is typically expressed in real numbers and floating-point arithmetic, many embedded and hardware-oriented implementations use **fixed-point arithmetic** to achieve efficient, deterministic, and low-power operation. In fixed-point representations, real-valued quantities are mapped to integers by assuming a fixed binary point position. For example, a real number $r$ might be represented as an integer $q$ such that:

$$
r \approx \frac{q}{2^F}
$$

where $F$ is the number of fractional bits.

Fixed-point arithmetic introduces quantization effects and requires explicit handling of scaling, rounding, and saturation. This specification defines a neuron model at the conceptual level that:

- represents inputs, weights, bias, and output as signed fixed-point values (two's-complement integers with an implicit binary point),
- defines how binary-point alignment is performed (especially for the bias),
- defines a deterministic rounding rule when reducing precision,
- allows for common activation-function choices in the fixed-point domain, and
- defines explicit output saturation.

---

## 2. Mathematical Behavior

### 2.1 Real-Valued Neuron (Conceptual)

Let:

- $N$ be the number of inputs,
- $x_0, x_1, \dots, x_{N-1}$ be real-valued inputs,
- $w_0, w_1, \dots, w_{N-1}$ be real-valued weights,
- $b$ be a real-valued bias,
- $m_0, m_1, \dots, m_{N-1}$ be a set of real-valued *mask* coefficients, typically $m_i \in \{0, 1\}$,
- $f_{\text{act}}(\cdot)$ be the activation function.

The **pre-activation** (masked weighted sum plus bias) is:

$$
s = b + \sum_{i=0}^{N-1} m_i\, x_i \cdot w_i.
$$

Here:

- $m_i = 1$ means the corresponding input–weight product is used,
- $m_i = 0$ means the corresponding contribution is completely skipped,
- other values $m_i \in [0,1]$ could represent partial attenuation (though the most common case is a binary mask).

The **post-activation** output is:

$$
y_{\text{real}} = f_{\text{act}}(s).
$$

Typical choices for $f_{\text{act}}$ include:

- **Identity** (linear neuron):

$$
f_{\text{act}}(s) = s
$$

- **ReLU**:

$$
f_{\text{act}}(s) = \max(0, s)
$$

- **Leaky ReLU** (with some small slope $\alpha \in (0,1)$ for negative values):

$$
f_{\text{act}}(s) =
\begin{cases}
\alpha\, s, & \text{if } s < 0 \\
s, & \text{otherwise}
\end{cases}
$$

- **Hard-tanh**–style clamp (bounded output, often approximating $\tanh$):

$$
f_{\text{act}}(s) =
\begin{cases}
-1, & \text{if } s < -1 \\
s, & \text{if } -1 \le s \le 1 \\
+1, & \text{if } s > 1
\end{cases}
$$

Different activation choices change the expressive power and training dynamics of networks built from these neurons. For example:

- Identity keeps the neuron purely linear (useful in some layers or for debugging).
- ReLU encourages sparse activations by zeroing negative responses.
- Leaky ReLU mitigates "dead" neurons by allowing a small gradient for negative inputs.
- Hard-tanh keeps outputs in a controlled range, which can be desirable for numerical stability or when chaining many layers.

A mask vector $m$ introduces an additional form of sparsity at the level of connections: some input–weight pairs are effectively absent from the computation. Conceptually, this supports:

- pruning or structured sparsity (removing unimportant connections),
- conditional computation (selectively activating subsets of inputs),
- modeling architectures where only a subset of features should influence a specific neuron.

### 2.2 Fixed-Point Encoding

Each quantity is represented as a signed two's-complement integer with an implicit binary point:

- Input $x_i$ is represented by an integer $X_i$ with $F_x$ fractional bits:

$$
x_i \approx \frac{X_i}{2^{F_x}}
$$

- Weight $w_i$ is represented by an integer $W_i$ with $F_w$ fractional bits:

$$
w_i \approx \frac{W_i}{2^{F_w}}
$$

- Bias $b$ is represented by an integer $B$ with $F_b$ fractional bits:

$$
b \approx \frac{B}{2^{F_b}}
$$

- Output $y$ is represented by an integer $Y$ with $F_y$ fractional bits:

$$
y \approx \frac{Y}{2^{F_y}}
$$

A product $X_i \cdot W_i$ naturally has:

$$
F_p = F_x + F_w
$$

fractional bits. It is convenient to interpret the accumulator in this same "product scale" (i.e., with $F_p$ fractional bits).

A mask coefficient $m_i \in \{0,1\}$ is typically represented as an integer $M_i$ taking values 0 (skip) or 1 (use), so that:

$$
m_i\, x_i w_i \approx \frac{M_i\, X_i W_i}{2^{F_p}}.
$$

### 2.3 Fixed-Point Computation (Integer Domain)

The computation is defined in integer form while tracking the implied binary-point positions.

#### 2.3.1 Bias Alignment (to product scale)

The bias integer $B$ (with $F_b$ fractional bits) is aligned into the accumulator scale (with $F_p$ fractional bits), producing $B_{\text{aligned}}$.

- If $F_b > F_p$, shift right by $sh = F_b - F_p$ with rounding (rule below).
- If $F_p > F_b$, shift left by $sh = F_p - F_b$.
- If $F_b = F_p$, no shift is required.

**Deterministic rounding rule for right shifts** (used whenever shifting right by $sh \ge 1$):

1. Compute $c = 2^{sh-1}$.
2. If the value is nonnegative, add $c$ before shifting.
3. If the value is negative, subtract $c$ before shifting.
4. Then perform an arithmetic right shift by $sh$.

This rule is intentionally stated as an algorithm (rather than labeled as "round-to-nearest"), since its behavior for negative values depends on two's-complement arithmetic shifting.

#### 2.3.2 Multiply–Accumulate with Mask

Introduce mask integers $M_i \in \{0,1\}$, which represent the fixed-point version of the real-valued masks $m_i$. These are used to selectively include or exclude each input–weight product from the accumulation.

Initialize the integer accumulator:

$$
ACC_0 = B_{\text{aligned}}
$$

Then accumulate masked products:

$$
ACC_{k+1} = ACC_k + M_k \cdot (X_k \cdot W_k), \quad k = 0,1,\dots,N-1.
$$

After all inputs:

$$
ACC_{\text{final}} = B_{\text{aligned}} + \sum_{i=0}^{N-1} M_i \cdot (X_i \cdot W_i).
$$

The case $M_i = 0$ effectively removes the corresponding product from the sum, while $M_i = 1$ keeps it unchanged. Conceptually, this captures connection-level sparsity in the fixed-point domain.

**Product correctness requirement:** each included term $(X_i \cdot W_i)$ must equal the signed two's-complement integer multiplication of the encoded operands, producing the full signed product width $(W_x + W_w)$ bits (before any later widening/sign-extension for accumulation). The multiplication may be implemented combinationally or computed over multiple cycles (sequentially); only the arithmetic result matters.

#### 2.3.3 Activation in the Accumulator Scale

An activation function is applied to $ACC_{\text{final}}$ while it is still interpreted in the accumulator's fixed-point scale (with $F_p$ fractional bits). For each activation type, the fixed-point behavior is defined to mirror the real-valued forms in Section 2.1.

Let $ACC_{\text{act}}$ denote the accumulator after activation. The following common activation choices are considered:

- **Identity (no nonlinearity)**:

$$
ACC_{\text{act}} = ACC_{\text{final}}
$$

- **ReLU** (clamping negative values to zero):

$$
ACC_{\text{act}} =
\begin{cases}
0, & \text{if } ACC_{\text{final}} < 0 \\
ACC_{\text{final}}, & \text{otherwise}
\end{cases}
$$

- **Leaky ReLU** (fixed negative slope):

  Choose a constant slope $\alpha \in (0,1)$. In hardware-oriented fixed-point implementations, $\alpha$ is often chosen as a reciprocal power of two, e.g. $\alpha = 1 / 2^L$, so that scaling by $\alpha$ corresponds to an arithmetic right shift by $L$ bits.

  Conceptually, the integer-domain behavior is:

$$
ACC_{\text{act}} =
\begin{cases}
\left\lfloor \alpha \cdot ACC_{\text{final}} \right\rfloor, & \text{if } ACC_{\text{final}} < 0 \\
ACC_{\text{final}}, & \text{otherwise}
\end{cases}
$$

  where "$\left\lfloor \cdot \right\rfloor$" denotes an appropriate fixed-point rounding or truncation.

- **Hard-tanh–style clamp** (bounded interval in accumulator scale):

  In the real-valued formulation, the hard-tanh activation clamps the value to the interval $[-1, +1]$. In the accumulator's fixed-point representation (with fractional bits $F_p$), the integers corresponding to $-1$ and $+1$ are:

$$
L_{\min} = -2^{F_p}, \quad L_{\max} = +2^{F_p}.
$$

  The activation in the accumulator domain is then:

$$
ACC_{\text{act}} =
\begin{cases}
L_{\max}, & \text{if } ACC_{\text{final}} > L_{\max} \\
L_{\min}, & \text{if } ACC_{\text{final}} < L_{\min} \\
ACC_{\text{final}}, & \text{otherwise}
\end{cases}
$$

These activation modes are intended to be selectable (for example, per neuron or per operation) to cover a range of behaviors commonly used in neural-network practice, from purely linear to strongly nonlinear and bounded.

#### 2.3.4 Output Quantization (product scale → output scale)

After activation, the accumulator $ACC_{\text{act}}$ is converted from the product scale (with $F_p$ fractional bits) to the output scale (with $F_y$ fractional bits). This is analogous to the bias-alignment step, reusing the same deterministic rounding rule for right shifts.

- If $F_p > F_y$, shift right by $sh = F_p - F_y$ using the deterministic rounding rule above.
- If $F_y > F_p$, shift left by $sh = F_y - F_p$.
- If $F_p = F_y$, no shift.

Let the quantized integer be $Q$.

#### 2.3.5 Output Saturation (finite signed range)

The output integer $Y$ is obtained by saturating $Q$ to the signed range representable by the chosen output width $W_y$:

$$
Y_{\min} = -2^{W_y-1}, \quad Y_{\max} = 2^{W_y-1}-1
$$

$$
Y =
\begin{cases}
Y_{\max}, & \text{if } Q > Y_{\max} \\
Y_{\min}, & \text{if } Q < Y_{\min} \\
Q, & \text{otherwise}
\end{cases}
$$

This is the only stage where explicit saturation to the final output width is required; internal accumulations may wrap or be wider, depending on implementation choices.

---

## 3. Sequential Signed Multiplier Submodule (Conceptual)

Some implementations factor multiplication into a dedicated signed multiplier block that can compute a product over multiple cycles (e.g., to reduce area). This section specifies the *functional* behavior of such a multiplier submodule, independent of any particular internal algorithm.

### 3.1 Inputs, Outputs, and Arithmetic Meaning

Let:

- input operand $a$ be a signed two's-complement integer of width $A_W$,
- input operand $b$ be a signed two's-complement integer of width $B_W$.

The multiplier must produce:

- $p = a \cdot b$ as a signed two's-complement integer of width $P_W = A_W + B_W$.

Because $P_W$ is the full mathematical product width for two signed integers of widths $A_W$ and $B_W$, the result must be exact in that width (i.e., no truncation is needed to represent $a \cdot b$ in $P_W$ bits).

### 3.2 Transaction-Level Behavior (Conceptual Handshake)

The multiplier operates as a transactional unit:

- It **accepts** a new pair $(a,b)$ when both:
  - the caller indicates an input is valid, and
  - the multiplier indicates it is ready.
- After accepting an input pair, the multiplier may take **multiple cycles** before producing the result.
- When the result is available, it indicates output-valid and presents $p$.
- If the environment is not ready to accept the result, the multiplier must be able to **hold** the result stable while output-valid remains asserted, until the result is accepted.

This description intentionally avoids prescribing any exact cycle count or micro-architecture; it only defines the observable input/output contract of a multi-cycle multiplier.

### 3.3 Accept/Produce Ordering

A multiplier instance must behave like a single in-flight operation engine:

- It must not accept a second input pair if it already has an unconsumed output result pending.
- Each accepted input pair must produce exactly one output product, in the same order inputs were accepted.

### 3.4 Implementation Note (Non-Normative)

A common way to implement a multi-cycle multiplier is a **shift-and-add** scheme over the multiplier operand bits, combined with sign handling (e.g., multiply magnitudes and apply the final sign). This is offered as intuition only; any method is allowed so long as Sections 3.1–3.3 are satisfied.

---

## 4. Bit-Width and Overflow Notes (Implementation-Relevant)

This section captures common fixed-point implementation consequences when using finite-width signed arithmetic.

### 4.1 Typical width growth

If inputs use $W_x$ bits and weights use $W_w$ bits (both signed), the full-precision product typically needs:

$$
W_p = W_x + W_w
$$

bits (signed). When summing $N$ products, a common rule-of-thumb for additional headroom is approximately $\lceil \log_2(N) \rceil$ bits, plus any explicit guard bits to reduce wraparound risk:

$$
W_{\text{acc}} \approx W_p + \left\lceil \log_2(N) \right\rceil + G
$$

where $G$ is a chosen number of guard bits.

A sparsity mask does not change the worst-case bound (since all masks could be 1), but in many practical workloads it reduces the *typical* magnitude of the sum because fewer products are included. This can be useful when selecting $G$ and other design parameters, although no assumption should rely solely on sparsity for correctness.

### 4.2 Wraparound vs saturation

In many hardware-friendly fixed-point designs:

- internal arithmetic is performed in a fixed width and may **wrap around** on overflow (two's-complement modular behavior), and  
- only the final output is **explicitly saturated** to the output width's signed range.

Practical implication: scaling choices (fractional bits), accumulator width, and activation behavior must be chosen so that internal wraparound is either extremely unlikely under expected workloads, or explicitly acceptable. Bounded activations such as hard-tanh can further reduce the dynamic range of intermediate values that must be carried into subsequent layers or stages.

---

## 5. Working Examples

### Example A — Fractional fixed-point, no scale changes

Assume:

- $N = 2$
- $F_x = 4$, $F_w = 4$ so $F_p = 8$
- $F_b = 8$ (bias already aligned to product scale)
- $F_y = 8$ (no output scale change)
- Activation chosen as ReLU (or any activation that leaves positive values unchanged in this range).
- All masks are 1: $M_0 = M_1 = 1$.

Real values:

- $x = [0.5, -1.25]$
- $w = [1.0, 0.5]$
- $b = 0.5$

Integer encodings:

- $X = [0.5 \cdot 2^4, -1.25 \cdot 2^4] = [8, -20]$
- $W = [1.0 \cdot 2^4, 0.5 \cdot 2^4] = [16, 8]$
- $B = 0.5 \cdot 2^8 = 128$

Integer-domain computation (accumulator interpreted at $F_p = 8$ fractional bits):

- Products: $8 \cdot 16 = 128$, and $-20 \cdot 8 = -160$.
- Accumulate (with full mask): $ACC = 128 + 128 - 160 = 96$.
- Activation: unchanged (nonnegative, so ReLU or identity leaves it at 96).
- Output scale: unchanged, so $Y = 96$.

Interpretation:

$$
y \approx \frac{96}{2^8} = 0.375
$$

which matches:

$$
0.5 + (0.5 \cdot 1.0) + (-1.25 \cdot 0.5) = 0.375.
$$

### Example B — Demonstrating deterministic rounding on a right shift

Assume a right shift by $sh = 2$ is needed when reducing fractional precision.

Let the integer value be $v = 5$ (in some fixed-point scale). The deterministic rounding rule does:

- $c = 2^{sh-1} = 2$
- $v \ge 0$, so $v' = v + c = 7$
- $v_{\text{out}} = v' \gg 2 = 1$

This example illustrates the "add/subtract $2^{sh-1}$ then arithmetic shift" rounding algorithm used whenever precision is reduced via right shifts.

### Example C — Effect of a sparsity mask

Keep the same scaling as in Example A ($F_x = F_w = 4$, $F_p = 8$, $F_b = F_y = 8$). Consider:

- $x = [0.5, -1.25]$
- $w = [1.0, 0.5]$
- $b = 0.5$

and two different mask settings:

1. **Full mask**: $m = [1, 1]$ (all contributions included).  
   This is exactly Example A above, yielding $y \approx 0.375$.

2. **Sparse mask**: $m = [1, 0]$ (second contribution disabled).  

   In the real domain:

$$
s = b + m_0 x_0 w_0 + m_1 x_1 w_1
  = 0.5 + 1 \cdot 0.5 \cdot 1.0 + 0 \cdot (-1.25 \cdot 0.5)
  = 0.5 + 0.5 + 0 = 1.0
$$

   With ReLU or identity activation and no saturation in this range, the output becomes:

$$
y_{\text{real}} = 1.0
$$

   In fixed-point terms, the second product is simply omitted from the accumulation, illustrating how a mask can remove selected input–weight contributions while leaving the rest unchanged.

---

## 6. Summary of Required Behavior

A compliant implementation, following this conceptual model, must:

1. Represent inputs, weights, bias, and output as signed fixed-point values (two's-complement integers with an implicit binary point).
2. Interpret products in a scale with fractional bits $F_p = F_x + F_w$ and accumulate in that same scale.
3. Align the bias into the product/accumulator scale before accumulation (left shift if increasing fractional bits, right shift if decreasing fractional bits), using the deterministic rounding rule for right shifts.
4. Optionally apply an element-wise sparsity mask $M_i \in \{0,1\}$ so that each input–weight product can be either included ($M_i = 1$) or excluded ($M_i = 0$) from the sum.
5. Compute each included product as a correct signed two's-complement multiplication at full product width $W_p = W_x + W_w$. The multiplication may be computed sequentially over multiple cycles; only the arithmetic result is specified here.
6. Apply an activation function in the accumulator domain, selectable among:
   - identity,
   - ReLU (clamp negative values to zero),
   - leaky ReLU (scale negative values by a fixed factor $\alpha \in (0,1)$, often a reciprocal power of two in fixed-point), or
   - hard-tanh–style clamp to a bounded interval (e.g. the integers corresponding to $[-1, +1]$ in the accumulator scale).
7. Quantize from the accumulator's fractional scale to the output's fractional scale using the same shift-and-round rule for right shifts, then saturate the result to the signed output range representable by the chosen output width.

Additionally, if a dedicated sequential signed multiplier submodule is used, it must satisfy the functional contract in Section 3 (exact signed product at width $A_W+B_W$, multi-cycle allowed, transactional accept/produce behavior, and stable output while pending acceptance).
