# neuron_mac_serial — RTL Code Completion Task (Fixed-Point Neuron MAC)

## 1. Why I chose this task

I chose this task because it is a realistic small but nontrivial RTL unit that forces an agent to reason across:
- fixed-point scaling and deterministic rounding,
- multi-cycle sequencing (serial MAC across `NUM_INPUTS`),
- correctness under masking (sparsity) and selectable activations,
- integration of a multi-cycle **sequential signed multiplier submodule** via a valid/ready-style handshake,
- a simple ready/valid handshake contract for starting an operation and producing a result.

It’s compact enough to fit as a single-module code-completion problem, but still difficult in the ways RTL often is: cycle-accurate behavior, signed arithmetic corner cases, and fixed-point quantization/saturation rules.

## 2. Industry relevance

This design mirrors common building blocks used in practical hardware/embedded ML inference and DSP pipelines:
- **Fixed-point compute** is a default choice for resource/latency/power efficiency in FPGA/ASIC accelerators.
- **Serial MAC** maps well to resource-constrained implementations (area-first designs) and is a frequent baseline micro-architecture.
- **Sparsity masking** reflects real workloads (pruning, structured sparsity, conditional compute / gating).
- **Activation selection** (identity / ReLU / leaky ReLU / clamp) is representative of typical inference datapaths and bounded-activation stabilization patterns.
- **Saturation + deterministic rounding** are exactly the kinds of implementation details that cause real silicon/FPGA mismatches if not handled precisely.
- **Multi-cycle multiplier integration** (handshake, sequencing, and correctness) is a common real-world integration pain point when composing larger datapaths from reusable arithmetic blocks.

## 3. Brief context of the codebase

This repository is organized as a self-contained RTL task with:
- a conceptual specification describing the neuron math and fixed-point rules,
- a Verilog top module to complete (and a provided multiplier submodule to integrate),
- a cocotb-based test suite with a bit-accurate Python reference model.

### 3.1 Repository layout (recommended)

- `docs/Specification.md`  
  Conceptual spec: fixed-point encoding, bias alignment, masking, activation behavior, quantization (with deterministic right-shift rounding), and output saturation. It also defines the functional contract for the sequential signed multiplier submodule.

- `sources/neuron_mac_serial.v`  
  **Task RTL** (code completion): Verilog-2001 module `neuron_mac_serial` with the missing implementation region. The agent must instantiate and use the provided multiplier module.

- `sources/seq_mult_signed.v`  
  Provided **sequential signed multiplier** RTL (Verilog-2001). The top module must use it to compute `x[i]*w[i]` when `mask[i]=1`, respecting its input/output handshake.

- `test/test_neuron_mac_serial_hidden.py`  
  cocotb tests + a bit-accurate Python model (`model_serial`) used to validate functionality across:
  - directed fixed-point cases,
  - saturation cases,
  - reset/handshake behavior,
  - randomized regression,
  - activation modes + sparsity masking.

- `prompt.txt` (or `docs/Prompt.txt`)  
  The exact prompt given to the agent (ports, packing, handshake expectations, and “what to implement” guidance).

> Notes:
> - The testbench assumes little-endian packing for `x_flat` / `w_flat`: element 0 is in LSBs, element `i` is slice `i*W +: W`.
> - Input handshake requirement: `in_ready` must be high exactly when the block is idle and able to accept a new operation; the design must only capture inputs on `in_valid && in_ready`.
> - Output handshake: the interface includes `out_ready`, but the evaluation does not exercise output backpressure (you may assume `out_ready` is asserted whenever `out_valid` is asserted). The design should still produce a clean result indication via `out_valid` and return to idle after producing the result.

### 3.2 How to run locally (cocotb Makefile flow)

I ran the cocotb tests locally using a dedicated Makefile that targets Icarus Verilog and supports waveform dumping via `WAVES=1`.

Example Makefile:

```make
# Makefile.serial
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

# Include both the top module and the sequential multiplier module
VERILOG_SOURCES = \
  $(PWD)/sources/neuron_mac_serial.v \
  $(PWD)/sources/seq_mult_signed.v

TOPLEVEL = neuron_mac_serial
MODULE = test_neuron_mac_serial_hidden

include $(shell cocotb-config --makefiles)/Makefile.sim
````

Run command (with waveform dumping enabled):

```bash
make -f Makefile.serial SIM=icarus WAVES=1
```

## 4. Latest evaluation results

> IMPORTANT: HUD link must remain private (do not publish).

* **HUD run link (private):** `https://www.hud.ai/jobs/f545170d-2047-4752-82d3-efb87fcde735`
* **Model / config:** `claude-sonnet-4-5-20250929 --max-steps 150`
* **Score / Pass@10:** `30%`
* **Notes:**

  * This version is tuned into the target difficulty band (Pass@10 >0% and ≤30%).
  * The current task requires the agent to integrate a provided **multi-cycle sequential multiplier submodule** (valid/ready handshake) into the serial MAC sequencing, which increases integration/controls complexity without changing the underlying math.
  * The prompt/spec focus on functional correctness and explicitly stated interface behavior; output backpressure is not exercised by the evaluator (i.e., `out_ready` can be assumed asserted whenever `out_valid` is asserted).

## 5. What the agent is expected to do (one-paragraph summary)

Given the conceptual spec in `docs/Specification.md`, the agent must implement a synthesizable Verilog-2001 serial neuron MAC that:

* latches packed inputs/weights/bias/mask and activation on `in_valid && in_ready`,
* walks indices `0..NUM_INPUTS-1` in order and accumulates only masked products,
* obtains each included product via the provided `seq_mult_signed` submodule, launching only when the multiplier is ready and waiting for a valid product before accumulating and advancing,
* applies the selected activation in the accumulator domain,
* converts from internal fractional scale to `OUT_FRAC` using the deterministic rounding rule,
* saturates to the signed `OUT_W` range,
* asserts `out_valid` with `out_data` when the result is ready, then returns idle (the environment assumes `out_ready` is asserted when `out_valid` is asserted).
