`timescale 1ns/1ps
// ============================================================
// seq_mult_signed.v  (Verilog-2001)
// Signed sequential shift-add multiplier with valid/ready handshake.
//
// Behavior:
//   - Accepts (a,b) when in_valid && in_ready
//   - Computes product over B_W cycles (shift-add over |b| bits)
//   - Produces p (A_W+B_W bits, signed) with out_valid
//   - Holds out_valid and p stable until out_ready
// ============================================================

module seq_mult_signed #(
    parameter integer A_W = 8,
    parameter integer B_W = 8
)(
    input  wire                      clk,
    input  wire                      rst_n,

    input  wire                      in_valid,
    output wire                      in_ready,
    input  wire signed [A_W-1:0]     a,
    input  wire signed [B_W-1:0]     b,

    output reg                       out_valid,
    input  wire                      out_ready,
    output reg  signed [A_W+B_W-1:0] p
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

    localparam integer P_W   = A_W + B_W;
    localparam integer CNT_W = (B_W <= 1) ? 1 : clog2(B_W);

    reg                  busy;
    reg                  sign;

    reg [A_W-1:0]         mag_a;
    reg [B_W-1:0]         mag_b;

    reg [P_W-1:0]         mcand;
    reg [B_W-1:0]         mult;
    reg [P_W-1:0]         acc;
    reg [CNT_W-1:0]       bit_cnt;

    // Ready when not computing and not holding an unconsumed output
    assign in_ready = (~busy) & (~out_valid);

    // Combinational next-acc (for the current multiplier LSB)
    wire [P_W-1:0] acc_add = acc + mcand;
    wire [P_W-1:0] acc_next = mult[0] ? acc_add : acc;

    // -------------------------
    // Main sequential logic
    // -------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy     <= 1'b0;
            out_valid<= 1'b0;
            p        <= {P_W{1'b0}};

            sign     <= 1'b0;
            mag_a    <= {A_W{1'b0}};
            mag_b    <= {B_W{1'b0}};
            mcand    <= {P_W{1'b0}};
            mult     <= {B_W{1'b0}};
            acc      <= {P_W{1'b0}};
            bit_cnt  <= {CNT_W{1'b0}};
        end else begin
            // Hold output until accepted
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end

            // Start a new multiply
            if (in_valid && in_ready) begin
                busy <= 1'b1;

                // Determine sign and magnitudes
                sign  <= a[A_W-1] ^ b[B_W-1];
                mag_a <= a[A_W-1] ? (~a + {{(A_W-1){1'b0}},1'b1}) : a;
                mag_b <= b[B_W-1] ? (~b + {{(B_W-1){1'b0}},1'b1}) : b;

                // Initialize shift-add datapath
                mcand   <= {{(P_W-A_W){1'b0}}, (a[A_W-1] ? (~a + {{(A_W-1){1'b0}},1'b1}) : a)};
                mult    <= (b[B_W-1] ? (~b + {{(B_W-1){1'b0}},1'b1}) : b);
                acc     <= {P_W{1'b0}};
                bit_cnt <= {CNT_W{1'b0}};
            end
            else if (busy) begin
                // One shift-add iteration per cycle (over B_W bits)
                acc   <= acc_next;
                mcand <= (mcand << 1);
                mult  <= (mult >> 1);

                if (bit_cnt == (B_W-1)) begin
                    // Done: form signed product and present output
                    busy <= 1'b0;

                    if (!out_valid) begin
                        // Convert magnitude to signed result
                        if (sign) p <= -$signed(acc_next);
                        else      p <=  $signed(acc_next);

                        out_valid <= 1'b1;
                    end
                end else begin
                    bit_cnt <= bit_cnt + 1'b1;
                end
            end
        end
    end

endmodule
