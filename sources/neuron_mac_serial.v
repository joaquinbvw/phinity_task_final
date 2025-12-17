timescale 1ns/1ps
// ============================================================
// neuron_mac_serial.v  (Verilog-2001)
// Serial dot-product neuron: y = sum(mask[i] ? x[i]*w[i] : 0) + bias
// Fixed-point with rounding + saturation, runtime-selectable activation.
// ============================================================

module neuron_mac_serial #(
    parameter integer NUM_INPUTS = 8,

    parameter integer X_W       = 8,   // input sample width (signed)
    parameter integer W_W       = 8,   // weight width (signed)
    parameter integer B_W       = 32,  // bias width (signed)
    parameter integer OUT_W     = 16,  // output width (signed)

    // Fixed-point fractional bits for each quantity.
    parameter integer X_FRAC    = 4,
    parameter integer W_FRAC    = 4,
    parameter integer B_FRAC    = 8,
    parameter integer OUT_FRAC  = 8,

    // Extra headroom in accumulator beyond sum of products.
    parameter integer GUARD_BITS = 2
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // Input handshake: present x_flat/w_flat/bias with in_valid=1.
    // Module asserts in_ready when it can accept a new neuron operation.
    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire signed [B_W-1:0]        bias,
    input  wire        [NUM_INPUTS*X_W-1:0] x_flat,
    input  wire        [NUM_INPUTS*W_W-1:0] w_flat,

    // New: runtime-selectable activation function
    //   act_sel = 2'b00 : identity (no non-linearity)
    //   act_sel = 2'b01 : ReLU
    //   act_sel = 2'b10 : leaky ReLU (slope 1/4 for negative)
    //   act_sel = 2'b11 : hard-tanh style clamp to [-1.0, +1.0] in FRAC_P scale
    input  wire        [1:0]            act_sel,

    // New: sparsity mask (one bit per input element, 1 = use x[i]*w[i], 0 = skip)
    input  wire        [NUM_INPUTS-1:0] mask_flat,

    // Output handshake: out_valid pulses for 1 cycle with out_data.
    output reg                          out_valid,
    output reg  signed [OUT_W-1:0]      out_data,

    output reg                          busy
);

    // -------------------------
    // Small utility: clog2
    // -------------------------
    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            clog2 = 0;
            while (v > 0) begin
                clog2 = clog2 + 1;
                v = v >> 1;
            end
        end
    endfunction

    localparam integer PROD_W  = X_W + W_W;
    localparam integer FRAC_P  = X_FRAC + W_FRAC;

    // Accumulator width: product width + log2(NUM_INPUTS) + guard bits
    localparam integer SUM_GROW = (NUM_INPUTS <= 1) ? 1 : clog2(NUM_INPUTS);
    localparam integer ACC_W    = PROD_W + SUM_GROW + GUARD_BITS;

    // Counter/index width (at least 1)
    localparam integer CNT_W = (NUM_INPUTS <= 1) ? 1 : clog2(NUM_INPUTS);

    // Activation selector encoding (local use)
    localparam [1:0] ACT_IDENTITY = 2'b00;
    localparam [1:0] ACT_RELU     = 2'b01;
    localparam [1:0] ACT_LEAKY    = 2'b10;
    localparam [1:0] ACT_CLAMP    = 2'b11;

    // Leaky ReLU negative slope = 1/4 -> arithmetic shift right by 2
    localparam integer LEAK_SHIFT = 2;

    // -------------------------
    // Saturate ACC_W -> OUT_W
    // -------------------------
    function signed [OUT_W-1:0] sat_to_out;
        input signed [ACC_W-1:0] v;
        reg   signed [ACC_W-1:0] max_v;
        reg   signed [ACC_W-1:0] min_v;
        begin
            // Max:  0x7FF.. , Min: 0x800..
            max_v = $signed({1'b0, {(OUT_W-1){1'b1}}});
            min_v = $signed({1'b1, {(OUT_W-1){1'b0}}});

            if (v > max_v)       sat_to_out = {1'b0, {(OUT_W-1){1'b1}}};
            else if (v < min_v)  sat_to_out = {1'b1, {(OUT_W-1){1'b0}}};
            else                 sat_to_out = v[OUT_W-1:0];
        end
    endfunction

    // -------------------------
    // Align bias into ACC scale (FRAC_P fractional bits)
    // Includes rounding when shifting right.
    //
    // NOTE: If B_W > ACC_W, the assignment truncates to ACC_W bits
    // (this is intentional and is mirrored in the Python model).
    // -------------------------
    function signed [ACC_W-1:0] align_bias;
        input signed [B_W-1:0] b;
        reg   signed [ACC_W-1:0] be;
        reg   signed [ACC_W-1:0] round_const;
        integer sh;
        begin
            // Truncate or sign-extend into ACC_W
            be = b;

            if (B_FRAC > FRAC_P) begin
                sh = (B_FRAC - FRAC_P);          // right shift amount (>=1)
                round_const = $signed(1) <<< (sh-1);
                if (be >= 0) be = be + round_const;
                else         be = be - round_const;
                align_bias = be >>> sh;
            end else if (FRAC_P > B_FRAC) begin
                sh = (FRAC_P - B_FRAC);
                align_bias = be <<< sh;
            end else begin
                align_bias = be;
            end
        end
    endfunction

    // -------------------------
    // Activation + quantize ACC (FRAC_P) -> OUT (OUT_FRAC),
    // then saturate to OUT_W. Uses act_sel (runtime).
    // -------------------------
    function signed [OUT_W-1:0] quantize_and_sat;
        input signed [ACC_W-1:0] vin;
        input        [1:0]       act_sel_f;

        reg   signed [ACC_W-1:0] v;
        reg   signed [ACC_W-1:0] round_const;
        reg   signed [ACC_W-1:0] clamp_pos;
        reg   signed [ACC_W-1:0] clamp_neg;
        integer sh;
        begin
            v = vin;

            // Activation operates in accumulator domain (FRAC_P fractional bits).
            // For ACT_CLAMP, clamp to [-1.0, +1.0] expressed in FRAC_P scale:
            //   1.0 -> 1 << FRAC_P
            clamp_pos = $signed(1) <<< FRAC_P;
            clamp_neg = -clamp_pos;

            case (act_sel_f)
                ACT_RELU: begin
                    // ReLU
                    if (v < 0) v = 0;
                end

                ACT_LEAKY: begin
                    // Leaky ReLU with slope 1/4 for negative side
                    if (v < 0) v = v >>> LEAK_SHIFT; // arithmetic shift
                end

                ACT_CLAMP: begin
                    // Hard-tanh style clamp to [-1.0, +1.0] in FRAC_P scale
                    if (v > clamp_pos)      v = clamp_pos;
                    else if (v < clamp_neg) v = clamp_neg;
                end

                default: begin
                    // ACT_IDENTITY (2'b00): no non-linearity
                end
            endcase

            // Adjust fractional bits from FRAC_P to OUT_FRAC
            if (FRAC_P > OUT_FRAC) begin
                sh = (FRAC_P - OUT_FRAC);       // right shift amount (>=1)
                round_const = $signed(1) <<< (sh-1);
                if (v >= 0) v = v + round_const;
                else        v = v - round_const;
                v = v >>> sh;                   // arithmetic shift
            end else if (OUT_FRAC > FRAC_P) begin
                sh = (OUT_FRAC - FRAC_P);
                v = v <<< sh;
            end

            quantize_and_sat = sat_to_out(v);
        end
    endfunction

    // -------------------------
    // Serial MAC datapath (indexed, not shifting)
    // -------------------------

    // Latched input vectors and sparsity mask
    reg [NUM_INPUTS*X_W-1:0] x_reg;
    reg [NUM_INPUTS*W_W-1:0] w_reg;
    reg [NUM_INPUTS-1:0]     mask_reg;

    // Latched activation selector (per operation)
    reg [1:0]                act_sel_reg;

    // Accumulator and element index
    reg signed [ACC_W-1:0]   acc;
    reg [CNT_W-1:0]          idx;

    // Current element (little-endian packing by index)
    wire signed [X_W-1:0] x_i;
    wire signed [W_W-1:0] w_i;
    wire                   mask_i;

    // Dynamic part-select: x[i], w[i], mask[i]
    assign x_i   = $signed(x_reg[idx*X_W +: X_W]);
    assign w_i   = $signed(w_reg[idx*W_W +: W_W]);
    assign mask_i = mask_reg[idx];

    // Product and masked version in accumulator width
    wire signed [PROD_W-1:0] prod        = x_i * w_i;
    wire signed [ACC_W-1:0]  prod_acc    = prod;  // sign-extend
    wire signed [ACC_W-1:0]  masked_prod = mask_i ? prod_acc
                                                 : {ACC_W{1'b0}};
    wire signed [ACC_W-1:0]  acc_next    = acc + masked_prod;

    // Ready is simply the complement of busy (combinational)
    assign in_ready = ~busy;

    // -------------------------
    // Main sequential logic
    // -------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy       <= 1'b0;
            out_valid  <= 1'b0;
            out_data   <= {OUT_W{1'b0}};
            x_reg      <= {NUM_INPUTS*X_W{1'b0}};
            w_reg      <= {NUM_INPUTS*W_W{1'b0}};
            mask_reg   <= {NUM_INPUTS{1'b0}};
            act_sel_reg<= 2'b00;
            acc        <= {ACC_W{1'b0}};
            idx        <= {CNT_W{1'b0}};
        end else begin
            // out_valid is a one-cycle pulse
            out_valid <= 1'b0;

            // Accept a new operation only when idle and in_valid is high
            if (in_valid && in_ready) begin
                busy        <= 1'b1;
                idx         <= {CNT_W{1'b0}};

                // Latch full input vectors (element 0 in LSBs) and mask/activation
                x_reg       <= x_flat;
                w_reg       <= w_flat;
                mask_reg    <= mask_flat;
                act_sel_reg <= act_sel;

                // Initialize accumulator with aligned bias (FRAC_P scale)
                acc         <= align_bias(bias);
            end
            else if (busy) begin
                // Consume one (possibly masked) x/w pair per cycle in index order
                acc <= acc_next;

                if (idx == (NUM_INPUTS-1)) begin
                    // Last element: produce result and return to idle
                    busy      <= 1'b0;
                    out_valid <= 1'b1;
                    out_data  <= quantize_and_sat(acc_next, act_sel_reg);
                end else begin
                    idx <= idx + 1'b1;
                end
            end
        end
    end

endmodule
