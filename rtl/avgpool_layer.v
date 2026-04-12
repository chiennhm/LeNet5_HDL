// ============================================================================
// Average-Pooling Layer Engine (2×2, stride 2) — Resource-Optimized
//
// Key changes vs. original:
//   1. All address computation uses sequential counters (no multipliers).
//   2. S_WAIT state for 1-cycle M4K dpram read latency.
//   3. Base-address registers (ch_base, row_base, pix_base) updated with
//      constant-only additions.
// ============================================================================
module avgpool_layer #(
    parameter IN_SIZE  = 24,
    parameter CHANNELS = 6
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,

    // Input read port
    output reg  [11:0] in_addr,
    input  wire signed [15:0] in_data,

    // Output write port
    output reg  [11:0] out_addr,
    output reg  signed [15:0] out_data,
    output reg         out_we
);

    localparam OUT_SIZE = IN_SIZE / 2;
    localparam POOL     = 2;

    // Address-increment constants (compile-time only)
    localparam [11:0] KY_STEP    = IN_SIZE - POOL + 1;  // delta when kx wraps
    localparam [11:0] IN_SIZE_X2 = IN_SIZE * 2;         // row stride in input
    localparam [11:0] IN_SIZE_SQ = IN_SIZE * IN_SIZE;   // channel stride

    // ---- Loop counters ----------------------------------------------------
    reg [7:0] ch, oy, ox;
    reg [1:0] ky, kx;

    // ---- Base-address registers (tracked cumulatively) --------------------
    reg [11:0] ch_base;    // start of current channel  (ch * IN²)
    reg [11:0] row_base;   // start of current row      (ch_base + oy*2*IN)
    reg [11:0] pix_base;   // start of current window   (row_base + ox*2)
    reg [11:0] out_cnt;    // sequential output counter

    // Accumulator (16-bit × 4 → 18 bits)
    reg signed [17:0] sum;

    // ---- FSM --------------------------------------------------------------
    localparam S_IDLE  = 3'd0,
               S_INIT  = 3'd1,
               S_WAIT  = 3'd2,
               S_ACC   = 3'd3,
               S_WRITE = 3'd4,
               S_DONE  = 3'd5;
    reg [2:0] state;

    // Average = sum / 4 = sum >>> 2
    wire signed [15:0] avg_val = sum[17:2];

    // ---- Main FSM ---------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            out_we   <= 1'b0;
            ch <= 0; oy <= 0; ox <= 0; ky <= 0; kx <= 0;
            sum      <= 18'sd0;
            ch_base  <= 12'd0;
            row_base <= 12'd0;
            pix_base <= 12'd0;
            out_cnt  <= 12'd0;
            in_addr  <= 12'd0;
            out_addr <= 12'd0;
            out_data <= 16'sd0;
        end else begin
            out_we <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        ch <= 0; oy <= 0; ox <= 0;
                        ch_base  <= 12'd0;
                        row_base <= 12'd0;
                        pix_base <= 12'd0;
                        out_cnt  <= 12'd0;
                        state    <= S_INIT;
                    end
                end

                S_INIT: begin
                    sum     <= 18'sd0;
                    ky <= 0; kx <= 0;
                    in_addr <= pix_base;
                    state   <= S_WAIT;
                end

                S_WAIT: begin
                    state <= S_ACC;
                end

                S_ACC: begin
                    sum <= sum + {{2{in_data[15]}}, in_data};

                    if (kx == POOL-1 && ky == POOL-1) begin
                        // Window complete
                        state <= S_WRITE;
                    end else begin
                        if (kx < POOL - 1) begin
                            kx      <= kx + 2'd1;
                            in_addr <= in_addr + 12'd1;
                        end else begin
                            kx <= 0;
                            ky <= ky + 2'd1;
                            in_addr <= in_addr + KY_STEP;
                        end
                        state <= S_WAIT;
                    end
                end

                S_WRITE: begin
                    out_addr <= out_cnt;
                    out_data <= avg_val;
                    out_we   <= 1'b1;
                    out_cnt  <= out_cnt + 12'd1;

                    if (ox < OUT_SIZE - 1) begin
                        ox       <= ox + 8'd1;
                        pix_base <= pix_base + 12'd2;   // stride = 2
                        state    <= S_INIT;
                    end else begin
                        ox <= 0;
                        if (oy < OUT_SIZE - 1) begin
                            oy       <= oy + 8'd1;
                            row_base <= row_base + IN_SIZE_X2;
                            pix_base <= row_base + IN_SIZE_X2;
                            state    <= S_INIT;
                        end else begin
                            oy <= 0;
                            if (ch < CHANNELS - 1) begin
                                ch       <= ch + 8'd1;
                                ch_base  <= ch_base  + IN_SIZE_SQ;
                                row_base <= ch_base  + IN_SIZE_SQ;
                                pix_base <= ch_base  + IN_SIZE_SQ;
                                state    <= S_INIT;
                            end else begin
                                state <= S_DONE;
                            end
                        end
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
