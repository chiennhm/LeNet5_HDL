// ============================================================================
// Convolution Layer Engine — Resource-Optimized for Cyclone II
//
// Key changes vs. original:
//   1. Weights stored in external weight_rom (M4K), accessed via w_addr/w_data.
//   2. Biases kept as small internal register array (INT8).
//   3. All address computation uses sequential counters (additions only,
//      no runtime multipliers).
//   4. Extra S_WAIT state accounts for 1-cycle M4K read latency.
//
// Arithmetic:
//   input  Q8.8 (16-bit signed)  ×  weight INT8 (Q0.7)
//   product = Q8.15 (24-bit)  →  accumulator Q16.15 (32-bit)
//   output  = acc[22:7] → Q8.8 with saturation + ReLU
// ============================================================================
module conv_layer #(
    parameter IN_SIZE     = 28,
    parameter IN_CH       = 1,
    parameter OUT_CH      = 6,
    parameter KERNEL      = 5,
    parameter BIAS_FILE   = "mem/conv1_bias.hex"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,

    // Input feature-map read port (to dpram)
    output reg  [11:0] in_addr,
    input  wire signed [15:0] in_data,

    // Output feature-map write port (to dpram)
    output reg  [11:0] out_addr,
    output reg  signed [15:0] out_data,
    output reg         out_we,

    // Weight ROM read port
    output reg  [14:0] w_addr,
    input  wire signed [7:0]  w_data
);

    // ---- Derived constants (all compile-time, no hardware) -----------------
    localparam OUT_SIZE   = IN_SIZE - KERNEL + 1;
    localparam FILT_SIZE  = IN_CH * KERNEL * KERNEL;   // weights per filter

    // Address-increment constants (additions only at runtime)
    localparam [11:0] KY_STEP = IN_SIZE - KERNEL + 1;
    //   when kx wraps → 0 and ky increments
    localparam [11:0] IC_STEP = IN_SIZE * IN_SIZE
                              - (KERNEL - 1) * (IN_SIZE + 1);
    //   when kx,ky both wrap and ic increments
    localparam [11:0] OY_STEP = KERNEL;
    //   pix_base delta when ox wraps → 0 and oy increments
    //   (= IN_SIZE - OUT_SIZE + 1 = KERNEL)

    // ---- Bias ROM (small, remains in LUT registers) -----------------------
    reg signed [7:0] biases [0:OUT_CH-1];
    initial $readmemh(BIAS_FILE, biases);

    // ---- Loop counters ----------------------------------------------------
    reg [7:0] oc, oy, ox, ic, ky, kx;

    // ---- Address / accumulator registers ----------------------------------
    reg [11:0] pix_base;       // input base addr for current (oy,ox)
    reg [14:0] w_base;         // weight base addr for current oc
    reg [11:0] out_cnt;        // sequential output address counter
    reg signed [31:0] acc;

    // ---- FSM --------------------------------------------------------------
    localparam S_IDLE  = 3'd0,
               S_BIAS  = 3'd1,
               S_WAIT  = 3'd2,   // 1-cycle pipeline for M4K read
               S_MAC   = 3'd3,
               S_WRITE = 3'd4,
               S_DONE  = 3'd5;
    reg [2:0] state;

    // ---- MAC arithmetic: Q8.8 × INT8(Q0.7) = Q8.15 (24-bit) -------------
    wire signed [23:0] mult = in_data * w_data;

    // ---- Output: Q16.15 accumulator → Q8.8 with saturation + ReLU --------
    wire signed [15:0] acc_q88 = acc[22:7];
    wire overflow = (acc[31:23] != {9{acc[22]}});
    wire signed [15:0] saturated = overflow
        ? (acc[31] ? 16'sh8000 : 16'sh7FFF)
        : acc_q88;
    wire signed [15:0] relu_out = saturated[15] ? 16'sh0000 : saturated;

    // ---- Main FSM ---------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            out_we   <= 1'b0;
            oc <= 0; oy <= 0; ox <= 0;
            ic <= 0; ky <= 0; kx <= 0;
            acc      <= 32'sd0;
            pix_base <= 12'd0;
            w_base   <= 15'd0;
            out_cnt  <= 12'd0;
            in_addr  <= 12'd0;
            w_addr   <= 15'd0;
            out_addr <= 12'd0;
            out_data <= 16'sd0;
        end else begin
            out_we <= 1'b0;   // default: no write

            case (state)
                // -----------------------------------------------------------
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        oc <= 0; oy <= 0; ox <= 0;
                        pix_base <= 12'd0;
                        w_base   <= 15'd0;
                        out_cnt  <= 12'd0;
                        state    <= S_BIAS;
                    end
                end

                // -----------------------------------------------------------
                S_BIAS: begin
                    // Load bias (INT8, Q0.7) into acc aligned to Q16.15
                    acc <= {{16{biases[oc][7]}}, biases[oc], 8'b0};
                    ic  <= 0; ky <= 0; kx <= 0;
                    // Issue first read addresses
                    in_addr <= pix_base;
                    w_addr  <= w_base;
                    state   <= S_WAIT;
                end

                // -----------------------------------------------------------
                S_WAIT: begin
                    // M4K registered read — data arrives next cycle
                    state <= S_MAC;
                end

                // -----------------------------------------------------------
                S_MAC: begin
                    // Accumulate product (sign-extend 24→32)
                    acc <= acc + {{8{mult[23]}}, mult};

                    if (kx == KERNEL-1 && ky == KERNEL-1 && ic == IN_CH-1) begin
                        // All products for this output pixel done
                        state <= S_WRITE;
                    end else begin
                        // Advance counters & set next addresses
                        if (kx < KERNEL - 1) begin
                            kx      <= kx + 8'd1;
                            in_addr <= in_addr + 12'd1;
                        end else begin
                            kx <= 0;
                            if (ky < KERNEL - 1) begin
                                ky      <= ky + 8'd1;
                                in_addr <= in_addr + KY_STEP;
                            end else begin
                                ky <= 0;
                                ic <= ic + 8'd1;
                                in_addr <= in_addr + IC_STEP;
                            end
                        end
                        w_addr <= w_addr + 15'd1;
                        state  <= S_WAIT;
                    end
                end

                // -----------------------------------------------------------
                S_WRITE: begin
                    out_addr <= out_cnt;
                    out_data <= relu_out;
                    out_we   <= 1'b1;
                    out_cnt  <= out_cnt + 12'd1;

                    // Advance output pixel (ox → oy → oc)
                    if (ox < OUT_SIZE - 1) begin
                        ox       <= ox + 8'd1;
                        pix_base <= pix_base + 12'd1;
                        state    <= S_BIAS;
                    end else begin
                        ox <= 0;
                        if (oy < OUT_SIZE - 1) begin
                            oy       <= oy + 8'd1;
                            pix_base <= pix_base + OY_STEP;
                            state    <= S_BIAS;
                        end else begin
                            oy       <= 0;
                            pix_base <= 12'd0;
                            if (oc < OUT_CH - 1) begin
                                oc     <= oc + 8'd1;
                                w_base <= w_base + FILT_SIZE;
                                state  <= S_BIAS;
                            end else begin
                                state <= S_DONE;
                            end
                        end
                    end
                end

                // -----------------------------------------------------------
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
