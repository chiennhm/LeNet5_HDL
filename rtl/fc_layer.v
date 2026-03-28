// ============================================================================
// Fully-Connected Layer Engine (Parameterized)
// Computes  out[j] = (Σ_i  w[j][i] * in[i]) + bias[j]   then optional ReLU
// Fixed-point Q8.8, 32-bit accumulator.
// ============================================================================
module fc_layer #(
    parameter IN_SIZE     = 120,
    parameter OUT_SIZE    = 84,
    parameter WEIGHT_FILE = "mem/fc1_weights.hex",
    parameter BIAS_FILE   = "mem/fc1_bias.hex",
    parameter APPLY_RELU  = 1            // 1 = apply ReLU, 0 = linear
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,

    // Input vector read port
    output wire [11:0] in_addr,
    input  wire signed [15:0] in_data,

    // Output vector write port
    output reg  [11:0] out_addr,
    output reg  signed [15:0] out_data,
    output reg         out_we
);

    localparam NUM_WEIGHTS = OUT_SIZE * IN_SIZE;
    localparam NUM_BIASES  = OUT_SIZE;

    // ---- Weight / Bias ROM ------------------------------------------------
    reg signed [15:0] weights [0:NUM_WEIGHTS-1];
    reg signed [15:0] biases  [0:NUM_BIASES-1];
    initial begin
        $readmemh(WEIGHT_FILE, weights);
        $readmemh(BIAS_FILE,   biases);
    end

    // ---- Counters ---------------------------------------------------------
    reg [7:0] j;    // output neuron index
    reg [7:0] i;    // input index

    // ---- Accumulator ------------------------------------------------------
    reg signed [31:0] acc;

    // ---- FSM --------------------------------------------------------------
    localparam S_IDLE  = 3'd0,
               S_BIAS  = 3'd1,
               S_MAC   = 3'd2,
               S_WRITE = 3'd3,
               S_DONE  = 3'd4;
    reg [2:0] state;

    // ---- Combinational input address --------------------------------------
    assign in_addr = {4'b0, i};

    // ---- Weight lookup ----------------------------------------------------
    wire [15:0] w_idx = j * IN_SIZE + i;
    wire signed [15:0] w_val = weights[w_idx];

    // ---- Multiply ---------------------------------------------------------
    wire signed [31:0] mult = in_data * w_val;

    // ---- Q16.16 → Q8.8 with saturation & optional ReLU -------------------
    wire signed [15:0] acc_q88 = acc[23:8];
    wire overflow = (acc[31:24] != {8{acc[23]}});
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
            out_addr <= 0;
            out_data <= 0;
        end else begin
            out_we <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        j <= 0;
                        state <= S_BIAS;
                    end
                end

                S_BIAS: begin
                    acc <= { {8{biases[j][15]}}, biases[j], 8'b0 };
                    i   <= 0;
                    state <= S_MAC;
                end

                S_MAC: begin
                    acc <= acc + mult;
                    if (i == IN_SIZE - 1) begin
                        state <= S_WRITE;
                    end else begin
                        i <= i + 1;
                    end
                end

                S_WRITE: begin
                    out_addr <= {4'b0, j};
                    out_data <= activated;
                    out_we   <= 1'b1;

                    if (j == OUT_SIZE - 1) begin
                        state <= S_DONE;
                    end else begin
                        j <= j + 1;
                        state <= S_BIAS;
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
