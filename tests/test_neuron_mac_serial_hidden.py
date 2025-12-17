import os
import random
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner


def twos(val, bits):
    return val & ((1 << bits) - 1)


def as_signed(val, bits):
    val = twos(val, bits)
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val


def sat_to_bits(v, out_w):
    mn = -(1 << (out_w - 1))
    mx = (1 << (out_w - 1)) - 1
    if v > mx:
        return mx
    if v < mn:
        return mn
    return v


def pack_list_signed(vals, w):
    out = 0
    for i, v in enumerate(vals):
        out |= (twos(v, w) << (i * w))
    return out


def pack_mask(mask_bits):
    """Pack list of 0/1 mask bits into little-endian bitfield."""
    out = 0
    for i, b in enumerate(mask_bits):
        if b:
            out |= (1 << i)
    return out


def clog2_py(value):
    v = value - 1
    c = 0
    while v > 0:
        c += 1
        v >>= 1
    return c if c > 0 else 1


def wrap_signed(v, bits):
    return as_signed(twos(v, bits), bits)


def fx_from_float(r, frac):
    # Inputs are already quantized by the testbench; keep it simple.
    # Prefer using values that are exact multiples of 2^-frac to avoid ambiguity.
    return int(round(r * (1 << frac)))


def fx_to_float(v, frac):
    return float(v) / float(1 << frac)


# Activation selector encodings (must match RTL)
ACT_IDENTITY = 0  # 2'b00
ACT_RELU     = 1  # 2'b01
ACT_LEAKY    = 2  # 2'b10
ACT_CLAMP    = 3  # 2'b11

LEAK_SHIFT   = 2  # same as RTL (slope = 1/4 for negative side)


def model_serial(
    x, w, bias,
    mask=None,
    act_sel=ACT_IDENTITY,
    NUM_INPUTS=8,
    X_W=8, W_W=8, B_W=32, OUT_W=16,
    X_FRAC=4, W_FRAC=4, B_FRAC=8, OUT_FRAC=8,
    GUARD_BITS=2,
):
    """
    Bit-accurate model of neuron_mac_serial.v including:
      - sparsity mask: sum(mask[i] ? x[i]*w[i] : 0)
      - bias alignment into FRAC_P = X_FRAC + W_FRAC (with rounding on right shifts)
      - serial accumulation with ACC_W wrap behavior
      - runtime activation (identity / ReLU / leaky ReLU / clamp)
      - output quantize to OUT_FRAC (with rounding on right shifts)
      - saturation to OUT_W

    NOTE: The new sequential multiplier + out_ready backpressure changes *timing*,
          not the arithmetic result, so this model remains unchanged.
    """

    if mask is None:
        mask = [1] * NUM_INPUTS

    PROD_W = X_W + W_W
    FRAC_P = X_FRAC + W_FRAC
    SUM_GROW = 1 if NUM_INPUTS <= 1 else clog2_py(NUM_INPUTS)
    ACC_W = PROD_W + SUM_GROW + GUARD_BITS

    def round_shift_right_acc(v_acc, sh):
        # Matches RTL:
        #   round_const = 1 << (sh-1)
        #   if v >= 0: v += round_const else v -= round_const
        #   v = v >>> sh
        v_acc = wrap_signed(v_acc, ACC_W)
        if sh <= 0:
            return v_acc
        round_const = 1 << (sh - 1)
        if v_acc >= 0:
            v_acc = wrap_signed(v_acc + round_const, ACC_W)
        else:
            v_acc = wrap_signed(v_acc - round_const, ACC_W)
        # Python >> is arithmetic for negative ints (matches >>> on signed)
        v_acc = wrap_signed(v_acc >> sh, ACC_W)
        return v_acc

    def align_bias(b):
        # RTL does: reg signed [ACC_W-1:0] be; be = b;  (may truncate if B_W > ACC_W)
        b_bits = twos(b, B_W)
        be_bits = b_bits & ((1 << ACC_W) - 1)
        be = as_signed(be_bits, ACC_W)

        if B_FRAC > FRAC_P:
            sh = B_FRAC - FRAC_P
            be = round_shift_right_acc(be, sh)
        elif FRAC_P > B_FRAC:
            sh = FRAC_P - B_FRAC
            be = wrap_signed(be << sh, ACC_W)

        return wrap_signed(be, ACC_W)

    def sat_to_out(v_acc):
        mx = (1 << (OUT_W - 1)) - 1
        mn = -(1 << (OUT_W - 1))
        if v_acc > mx:
            return mx
        if v_acc < mn:
            return mn
        return as_signed(twos(v_acc, OUT_W), OUT_W)

    def quantize_and_sat(vin_acc, act_sel_f):
        v = wrap_signed(vin_acc, ACC_W)

        clamp_pos = wrap_signed(1 << FRAC_P, ACC_W)        # +1.0
        clamp_neg = wrap_signed(-(1 << FRAC_P), ACC_W)     # -1.0

        if act_sel_f == ACT_RELU:
            if v < 0:
                v = 0
        elif act_sel_f == ACT_LEAKY:
            if v < 0:
                v = wrap_signed(v >> LEAK_SHIFT, ACC_W)
        elif act_sel_f == ACT_CLAMP:
            if v > clamp_pos:
                v = clamp_pos
            elif v < clamp_neg:
                v = clamp_neg
        else:
            pass

        if FRAC_P > OUT_FRAC:
            sh = FRAC_P - OUT_FRAC
            v = round_shift_right_acc(v, sh)
        elif OUT_FRAC > FRAC_P:
            sh = OUT_FRAC - FRAC_P
            v = wrap_signed(v << sh, ACC_W)

        return sat_to_out(v)

    acc = align_bias(bias)
    for i in range(NUM_INPUTS):
        xi = as_signed(x[i], X_W)
        wi = as_signed(w[i], W_W)
        if mask[i]:
            prod = xi * wi
            prod_s = as_signed(prod, PROD_W)
            acc = wrap_signed(acc + prod_s, ACC_W)

    return quantize_and_sat(acc, act_sel)


def read_signed(handle):
    """Compatible with cocotb versions that expose signed_integer or LogicArray.to_signed()."""
    v = handle.value
    if hasattr(v, "signed_integer"):
        return int(v.signed_integer)
    if hasattr(v, "to_signed"):
        return int(v.to_signed())
    return int(v)


# ----------------------------
# Parameter introspection helpers
# ----------------------------

def _get_param_int(dut, name, default):
    if hasattr(dut, name):
        obj = getattr(dut, name)
        try:
            return int(obj)
        except Exception:
            try:
                return int(obj.value)
            except Exception:
                return default
    return default


def get_dut_params(dut):
    params = {}
    params["NUM_INPUTS"] = _get_param_int(dut, "NUM_INPUTS", 8)
    params["X_W"]        = _get_param_int(dut, "X_W", 8)
    params["W_W"]        = _get_param_int(dut, "W_W", 8)
    params["B_W"]        = _get_param_int(dut, "B_W", 32)
    params["OUT_W"]      = _get_param_int(dut, "OUT_W", 16)
    params["X_FRAC"]     = _get_param_int(dut, "X_FRAC", 4)
    params["W_FRAC"]     = _get_param_int(dut, "W_FRAC", 4)
    params["B_FRAC"]     = _get_param_int(dut, "B_FRAC", 8)
    params["OUT_FRAC"]   = _get_param_int(dut, "OUT_FRAC", 8)
    params["GUARD_BITS"] = _get_param_int(dut, "GUARD_BITS", 2)
    return params


async def generate_clock(dut, period_ns=10):
    half = period_ns // 2
    while True:
        dut.clk.value = 0
        await Timer(half, unit="ns")
        dut.clk.value = 1
        await Timer(half, unit="ns")


async def reset_dut(dut, cycles=5):
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.bias.value = 0
    dut.x_flat.value = 0
    dut.w_flat.value = 0
    dut.act_sel.value = ACT_IDENTITY
    dut.mask_flat.value = 0

    # New: out_ready (default allow progress)
    if hasattr(dut, "out_ready"):
        dut.out_ready.value = 1

    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # After reset, out_valid should be low, idle/ready high
    assert int(dut.out_valid.value) == 0, "out_valid must be 0 after reset"
    assert int(dut.busy.value) == 0, "busy must be 0 after reset"
    assert int(dut.in_ready.value) == 1, "in_ready must be 1 after reset"


async def apply_and_check_one(
    dut, x, w, bias,
    mask=None,
    act_sel=ACT_IDENTITY,
    NUM_INPUTS=8,
    X_W=8, W_W=8, B_W=32, OUT_W=16,
    X_FRAC=4, W_FRAC=4, B_FRAC=8, OUT_FRAC=8,
    GUARD_BITS=2,
    max_wait_cycles=5000,
    stall_after_out_valid_cycles=0,
):
    """
    Drive one transaction and check result.

    New handshake behavior supported:
      - out_valid may stay asserted until out_ready is high.
      - while out_valid=1 and out_ready=0, out_data must remain stable.
    """
    if mask is None:
        mask = [1] * NUM_INPUTS

    dut.x_flat.value = pack_list_signed(x, X_W)
    dut.w_flat.value = pack_list_signed(w, W_W)
    dut.bias.value   = twos(bias, B_W)
    dut.mask_flat.value = pack_mask(mask)
    dut.act_sel.value = act_sel

    if hasattr(dut, "out_ready"):
        dut.out_ready.value = 1  # default: accept promptly unless we choose to stall later

    # Wait for in_ready
    for _ in range(max_wait_cycles):
        if int(dut.busy.value) == 1:
            assert int(dut.in_ready.value) == 0, "in_ready must be 0 while busy"
        else:
            assert int(dut.in_ready.value) == 1, "in_ready must be 1 when not busy"

        if int(dut.in_ready.value) == 1:
            break
        await RisingEdge(dut.clk)
    else:
        raise AssertionError("Timeout waiting for in_ready==1")

    # Pulse in_valid for one cycle
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0

    # Wait for out_valid to assert (not assuming one-cycle pulse anymore)
    got = None
    saw_out_valid = False

    for _ in range(max_wait_cycles):
        if int(dut.busy.value) == 1:
            assert int(dut.in_ready.value) == 0, "in_ready must be 0 while busy"
        else:
            assert int(dut.in_ready.value) == 1, "in_ready must be 1 when not busy"

        if int(dut.out_valid.value) == 1:
            saw_out_valid = True
            got = read_signed(dut.out_data)
            break

        await RisingEdge(dut.clk)

    if not saw_out_valid:
        raise AssertionError("Timeout waiting for out_valid==1")

    # Optional: apply backpressure and ensure out_data is stable while stalled
    if hasattr(dut, "out_ready") and stall_after_out_valid_cycles > 0:
        dut.out_ready.value = 0

        stable = got
        for _ in range(stall_after_out_valid_cycles):
            await RisingEdge(dut.clk)
            assert int(dut.out_valid.value) == 1, "out_valid must stay high while out_ready=0"
            assert read_signed(dut.out_data) == stable, "out_data must remain stable while stalled"
            assert int(dut.busy.value) == 1, "busy must remain high while output is pending"
            assert int(dut.in_ready.value) == 0, "in_ready must remain low while busy"

        dut.out_ready.value = 1

    # Complete the output handshake: wait for the DUT to return to idle
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.out_valid.value) == 0 and int(dut.busy.value) == 0:
            break
    else:
        raise AssertionError("Timeout waiting for DUT to drop out_valid/busy after out_ready")

    assert int(dut.in_ready.value) == 1, "in_ready must be 1 after result is accepted"

    # Compare against the model
    exp = model_serial(
        x, w, bias,
        mask=mask,
        act_sel=act_sel,
        NUM_INPUTS=NUM_INPUTS,
        X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
        GUARD_BITS=GUARD_BITS,
    )

    if got != exp:
        x_f = [fx_to_float(as_signed(v, X_W), X_FRAC) for v in x]
        w_f = [fx_to_float(as_signed(v, W_W), W_FRAC) for v in w]
        bias_f = fx_to_float(as_signed(bias, B_W), B_FRAC)
        got_f = fx_to_float(as_signed(got, OUT_W), OUT_FRAC)
        exp_f = fx_to_float(as_signed(exp, OUT_W), OUT_FRAC)
        raise AssertionError(
            "Mismatch:\n"
            f"  got={got} ({got_f})\n"
            f"  exp={exp} ({exp_f})\n"
            f"  x_fixed={x}\n"
            f"  w_fixed={w}\n"
            f"  bias_fixed={bias}\n"
            f"  mask={mask}\n"
            f"  act_sel={act_sel}\n"
            f"  x_real={x_f}\n"
            f"  w_real={w_f}\n"
            f"  bias_real={bias_f}\n"
        )


# ----------------------------
# Tests
# ----------------------------

@cocotb.test()
async def test_known_vector_fractional(dut):
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    p = get_dut_params(dut)
    NUM_INPUTS = p["NUM_INPUTS"]
    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
    GUARD_BITS = p["GUARD_BITS"]

    x_r = [0.5, -1.25, 2.0, -0.75, 1.5, 0.25, -2.5, 0.0]
    w_r = [1.0, 0.5, -1.5, 2.0, -0.25, 1.75, 0.5, -1.0]
    bias_r = 0.5

    x = [fx_from_float(v, X_FRAC) for v in x_r]
    w = [fx_from_float(v, W_FRAC) for v in w_r]
    bias = fx_from_float(bias_r, B_FRAC)
    mask = [1] * NUM_INPUTS

    await apply_and_check_one(
        dut, x, w, bias,
        mask=mask, act_sel=ACT_IDENTITY,
        NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
        GUARD_BITS=GUARD_BITS,
    )


@cocotb.test()
async def test_bias_only_fractional(dut):
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    p = get_dut_params(dut)
    NUM_INPUTS = p["NUM_INPUTS"]
    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
    GUARD_BITS = p["GUARD_BITS"]

    x = [0] * NUM_INPUTS
    w = [0] * NUM_INPUTS
    mask = [1] * NUM_INPUTS

    bias_r = 1.25
    bias = fx_from_float(bias_r, B_FRAC)

    await apply_and_check_one(
        dut, x, w, bias,
        mask=mask, act_sel=ACT_IDENTITY,
        NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
        GUARD_BITS=GUARD_BITS,
    )


@cocotb.test()
async def test_relu_clamp_to_zero_fractional(dut):
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    p = get_dut_params(dut)
    NUM_INPUTS = p["NUM_INPUTS"]
    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
    GUARD_BITS = p["GUARD_BITS"]

    x = [fx_from_float(-2.0, X_FRAC)] * NUM_INPUTS
    w = [fx_from_float(3.0, W_FRAC)] * NUM_INPUTS
    bias = fx_from_float(0.0, B_FRAC)
    mask = [1] * NUM_INPUTS

    await apply_and_check_one(
        dut, x, w, bias,
        mask=mask, act_sel=ACT_RELU,
        NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
        GUARD_BITS=GUARD_BITS,
    )


@cocotb.test()
async def test_positive_saturation(dut):
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    p = get_dut_params(dut)
    NUM_INPUTS = p["NUM_INPUTS"]
    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
    GUARD_BITS = p["GUARD_BITS"]

    max_x = (1 << (X_W - 1)) - 1
    max_w = (1 << (W_W - 1)) - 1
    x = [max_x] * NUM_INPUTS
    w = [max_w] * NUM_INPUTS
    bias = 0
    mask = [1] * NUM_INPUTS

    await apply_and_check_one(
        dut, x, w, bias,
        mask=mask, act_sel=ACT_IDENTITY,
        NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
        GUARD_BITS=GUARD_BITS,
    )


@cocotb.test()
async def test_back_to_back_transactions(dut):
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    p = get_dut_params(dut)
    NUM_INPUTS = p["NUM_INPUTS"]
    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
    GUARD_BITS = p["GUARD_BITS"]

    mask = [1] * NUM_INPUTS

    vecs = [
        (
            [fx_from_float(0.5, X_FRAC)] * NUM_INPUTS,
            [fx_from_float(0.5, W_FRAC)] * NUM_INPUTS,
            fx_from_float(0.0, B_FRAC),
        ),
        (
            [fx_from_float(1.0, X_FRAC), fx_from_float(-1.0, X_FRAC)] * (NUM_INPUTS // 2),
            [fx_from_float(1.5, W_FRAC)] * NUM_INPUTS,
            fx_from_float(0.25, B_FRAC),
        ),
    ]

    for x, w, bias in vecs:
        await apply_and_check_one(
            dut, x, w, bias,
            mask=mask, act_sel=ACT_IDENTITY,
            NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
            X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
            GUARD_BITS=GUARD_BITS,
        )


@cocotb.test()
async def test_sparsity_mask_basic(dut):
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    p = get_dut_params(dut)
    NUM_INPUTS = p["NUM_INPUTS"]
    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
    GUARD_BITS = p["GUARD_BITS"]

    x = [fx_from_float(1.0, X_FRAC)] * NUM_INPUTS
    w = [fx_from_float(0.5, W_FRAC)] * NUM_INPUTS
    bias = fx_from_float(0.0, B_FRAC)

    mask = [(i % 2 == 0) for i in range(NUM_INPUTS)]

    await apply_and_check_one(
        dut, x, w, bias,
        mask=mask, act_sel=ACT_IDENTITY,
        NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
        GUARD_BITS=GUARD_BITS,
    )


@cocotb.test()
async def test_activation_modes_sanity(dut):
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    p = get_dut_params(dut)
    NUM_INPUTS = p["NUM_INPUTS"]
    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
    GUARD_BITS = p["GUARD_BITS"]

    x = [fx_from_float(-1.0, X_FRAC)] * NUM_INPUTS
    w = [fx_from_float(0.75, W_FRAC)] * NUM_INPUTS
    bias = fx_from_float(-0.5, B_FRAC)
    mask = [1] * NUM_INPUTS

    for act_sel in (ACT_IDENTITY, ACT_RELU, ACT_LEAKY, ACT_CLAMP):
        await apply_and_check_one(
            dut, x, w, bias,
            mask=mask, act_sel=act_sel,
            NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
            X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
            GUARD_BITS=GUARD_BITS,
        )


#@cocotb.test()
#async def test_out_ready_backpressure(dut):
#    """New: verify out_valid holds and out_data is stable when out_ready is deasserted."""
#    cocotb.start_soon(generate_clock(dut))
#    await reset_dut(dut)
#
#    if not hasattr(dut, "out_ready"):
#        raise AssertionError("DUT is missing out_ready port, but test expects backpressure support")
#
#    p = get_dut_params(dut)
#    NUM_INPUTS = p["NUM_INPUTS"]
#    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
#    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
#    GUARD_BITS = p["GUARD_BITS"]
#
#    x = [fx_from_float(0.5, X_FRAC)] * NUM_INPUTS
#    w = [fx_from_float(-0.75, W_FRAC)] * NUM_INPUTS
#    bias = fx_from_float(0.25, B_FRAC)
#    mask = [1] * NUM_INPUTS
#
#    await apply_and_check_one(
#        dut, x, w, bias,
#        mask=mask, act_sel=ACT_IDENTITY,
#        NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
#        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
#        GUARD_BITS=GUARD_BITS,
#        stall_after_out_valid_cycles=5,
#    )


@cocotb.test()
async def test_random_regression_small(dut):
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    random.seed(2)

    p = get_dut_params(dut)
    NUM_INPUTS = p["NUM_INPUTS"]
    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
    GUARD_BITS = p["GUARD_BITS"]

    for _ in range(100):
        x = [random.randint(-32, 31) for _ in range(NUM_INPUTS)]
        w = [random.randint(-32, 31) for _ in range(NUM_INPUTS)]
        bias = random.randint(-512, 511)
        mask = [random.randint(0, 1) for _ in range(NUM_INPUTS)]
        act_sel = random.randint(0, 3)

        await apply_and_check_one(
            dut, x, w, bias,
            mask=mask, act_sel=act_sel,
            NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
            X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
            GUARD_BITS=GUARD_BITS,
        )


@cocotb.test()
async def test_negative_saturation_no_relu(dut):
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    p = get_dut_params(dut)
    NUM_INPUTS = p["NUM_INPUTS"]
    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
    GUARD_BITS = p["GUARD_BITS"]

    x = [fx_from_float(-4.0, X_FRAC)] * NUM_INPUTS
    w = [fx_from_float(4.0, W_FRAC)] * NUM_INPUTS
    bias = fx_from_float(0.0, B_FRAC)
    mask = [1] * NUM_INPUTS

    exp = model_serial(
        x, w, bias,
        mask=mask, act_sel=ACT_IDENTITY,
        NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
        GUARD_BITS=GUARD_BITS,
    )
    out_min = -(1 << (OUT_W - 1))
    assert exp == out_min, f"Model did not saturate to OUT_MIN as expected: exp={exp}, OUT_MIN={out_min}"

    await apply_and_check_one(
        dut, x, w, bias,
        mask=mask, act_sel=ACT_IDENTITY,
        NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
        GUARD_BITS=GUARD_BITS,
    )


@cocotb.test()
async def test_async_reset_while_busy(dut):
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    p = get_dut_params(dut)
    NUM_INPUTS = p["NUM_INPUTS"]
    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
    GUARD_BITS = p["GUARD_BITS"]

    x = [fx_from_float(0.5, X_FRAC)] * NUM_INPUTS
    w = [fx_from_float(0.5, W_FRAC)] * NUM_INPUTS
    bias = fx_from_float(0.0, B_FRAC)
    mask = [1] * NUM_INPUTS

    dut.x_flat.value = pack_list_signed(x, X_W)
    dut.w_flat.value = pack_list_signed(w, W_W)
    dut.bias.value   = twos(bias, B_W)
    dut.mask_flat.value = pack_mask(mask)
    dut.act_sel.value = ACT_IDENTITY
    if hasattr(dut, "out_ready"):
        dut.out_ready.value = 1

    while int(dut.in_ready.value) == 0:
        await RisingEdge(dut.clk)

    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0

    for _ in range(3):
        await RisingEdge(dut.clk)

    assert int(dut.busy.value) == 1, "DUT should be busy before mid-op reset"

    dut.rst_n.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.busy.value) == 0, "busy must drop after reset"
    assert int(dut.out_valid.value) == 0, "out_valid must be 0 after reset"
    assert int(dut.in_ready.value) == 1, "in_ready must be 1 after reset"

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    await apply_and_check_one(
        dut, x, w, bias,
        mask=mask, act_sel=ACT_IDENTITY,
        NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
        X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
        GUARD_BITS=GUARD_BITS,
    )


@cocotb.test()
async def test_random_regression_full_range(dut):
    cocotb.start_soon(generate_clock(dut))
    await reset_dut(dut)

    random.seed(7)

    p = get_dut_params(dut)
    NUM_INPUTS = p["NUM_INPUTS"]
    X_W = p["X_W"]; W_W = p["W_W"]; B_W = p["B_W"]; OUT_W = p["OUT_W"]
    X_FRAC = p["X_FRAC"]; W_FRAC = p["W_FRAC"]; B_FRAC = p["B_FRAC"]; OUT_FRAC = p["OUT_FRAC"]
    GUARD_BITS = p["GUARD_BITS"]

    for _ in range(50):
        x = [random.randint(-(1 << (X_W - 1)), (1 << (X_W - 1)) - 1) for _ in range(NUM_INPUTS)]
        w = [random.randint(-(1 << (W_W - 1)), (1 << (W_W - 1)) - 1) for _ in range(NUM_INPUTS)]
        bias = random.randint(-(1 << (B_W - 1)), (1 << (B_W - 1)) - 1)
        mask = [random.randint(0, 1) for _ in range(NUM_INPUTS)]
        act_sel = random.randint(0, 3)

        await apply_and_check_one(
            dut, x, w, bias,
            mask=mask, act_sel=act_sel,
            NUM_INPUTS=NUM_INPUTS, X_W=X_W, W_W=W_W, B_W=B_W, OUT_W=OUT_W,
            X_FRAC=X_FRAC, W_FRAC=W_FRAC, B_FRAC=B_FRAC, OUT_FRAC=OUT_FRAC,
            GUARD_BITS=GUARD_BITS,
        )


def test_neuron_mac_serial_hidden_runner():
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent

    # IMPORTANT: include the new multiplier RTL file here
    sources = [
        proj_path / "sources" / "neuron_mac_serial.v",
        proj_path / "sources" / "seq_mult_signed.v",
    ]

    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="neuron_mac_serial",
        always=True,
    )
    runner.test(
        hdl_toplevel="neuron_mac_serial",
        test_module="test_neuron_mac_serial_hidden",
    )
