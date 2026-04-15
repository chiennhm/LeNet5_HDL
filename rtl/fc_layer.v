// ============================================================================
// Fully-Connected Layer Engine — Resource-Optimized for Cyclone II
//
// Key changes vs. original:
//   1. Weights are requested through an external memory interface
//      (w_req/w_addr + w_valid/w_data).
//   2. Biases kept as small INT8 register array.
//   3. Weight address uses sequential counter (no j*IN_SIZE multiplier).
//   4. S_WAIT state waits for a valid memory response.
//
// Arithmetic: same Q8.8 × INT8(Q0.7) scheme as conv_layer.
// ============================================================================
module fc_layer #(
    parameter IN_SIZE     = 120,
    parameter OUT_SIZE    = 84,
    parameter BIAS_FILE   = "mem/fc1_bias.hex",
    parameter APPLY_RELU  = 1
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,

    // Input vector read port
    output reg  [11:0] in_addr,
    input  wire signed [15:0] in_data,

    // Output vector write port
    output reg  [11:0] out_addr,
    output reg  signed [15:0] out_data,
    output reg         out_we,

    // External weight read port (e.g. SRAM bridge)
    output reg         w_req,
    output reg  [14:0] w_addr,
    input  wire signed [7:0]  w_data,
    input  wire        w_valid
);

    // ---- Bias ROM (small, remains in LUT registers) -----------------------
    reg signed [7:0] biases [0:OUT_SIZE-1];
    initial $readmemh(BIAS_FILE, biases);

    // ---- Counters ---------------------------------------------------------
    reg [7:0] j;    // output neuron index
    reg [7:0] i;    // input index

    // ---- Weight address tracking ------------------------------------------
    reg [14:0] w_base;   // start address for current output neuron j

    // ---- Accumulator ------------------------------------------------------
    reg signed [31:0] acc;

    // ---- FSM --------------------------------------------------------------
    localparam S_IDLE  = 3'd0,
               S_BIAS  = 3'd1,
               S_WAIT  = 3'd2,
               S_MAC   = 3'd3,
               S_WRITE = 3'd4,
               S_DONE  = 3'd5;
    reg [2:0] state;

    // ---- Latched weight from memory response ------------------------------
    reg signed [7:0] w_data_q;

    // ---- MAC: Q8.8 × INT8 = 24-bit signed --------------------------------
    wire signed [23:0] mult = in_data * w_data_q;

    // ---- Output: Q16.15 → Q8.8 with saturation & optional ReLU -----------
    wire signed [15:0] acc_q88 = acc[22:7];
    wire overflow = (acc[31:23] != {9{acc[22]}});
    wire signed [15:0] saturated = overflow
        ? (acc[31] ? 16'sh8000 : 16'sh7FFF)
        : acc_q88;
    wire signed [15:0] activated = (APPLY_RELU && saturated[15])
        ? 16'sh0000
        : saturated;

    // ---- Main FSM ---------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            out_we   <= 1'b0;
            j <= 0; i <= 0;
            acc      <= 32'sd0;
            w_base   <= 15'd0;
            in_addr  <= 12'd0;
            w_req    <= 1'b0;
            w_addr   <= 15'd0;
            w_data_q <= 8'sd0;
            out_addr <= 12'd0;
            out_data <= 16'sd0;
        end else begin
            out_we <= 1'b0;
            w_req  <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        j      <= 0;
                        w_base <= 15'd0;
                        state  <= S_BIAS;
                    end
                end

                S_BIAS: begin
                    acc     <= {{16{biases[j][7]}}, biases[j], 8'b0};
                    i       <= 0;
                    in_addr <= 12'd0;
                    w_addr  <= w_base;
                    w_req   <= 1'b1;
                    state   <= S_WAIT;
                end

                S_WAIT: begin
                    if (w_valid) begin
                        w_data_q <= w_data;
                        state    <= S_MAC;
                    end
                end

                S_MAC: begin
                    acc <= acc + {{8{mult[23]}}, mult};

                    if (i == IN_SIZE - 1) begin
                        state <= S_WRITE;
                    end else begin
                        i       <= i + 8'd1;
                        in_addr <= in_addr + 12'd1;
                        w_addr  <= w_addr  + 15'd1;
                        w_req   <= 1'b1;
                        state   <= S_WAIT;
                    end
                end

                S_WRITE: begin
                    out_addr <= {4'b0, j};
                    out_data <= activated;
                    out_we   <= 1'b1;

                    if (j == OUT_SIZE - 1) begin
                        state <= S_DONE;
                    end else begin
                        j      <= j + 8'd1;
                        w_base <= w_base + IN_SIZE;
                        state  <= S_BIAS;
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
