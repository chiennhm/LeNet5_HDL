// ============================================================================
// Convolution Layer Engine (Parameterized)
// Performs 2-D convolution with KERNEL×KERNEL filters.
// Reads input feature map via address/data port, writes output via write port.
// Weights & biases loaded from hex files. ReLU applied at output.
// Fixed-point Q8.8 arithmetic (16-bit signed).
// ============================================================================
module conv_layer #(
    parameter IN_SIZE     = 14,       // input spatial dimension (square)
    parameter IN_CH       = 1,        // input channels
    parameter OUT_CH      = 6,        // output channels (num filters)
    parameter KERNEL      = 5,        // kernel spatial dimension
    parameter WEIGHT_FILE = "/mem/conv1_weights.hex",
    parameter BIAS_FILE   = "/mem/conv1_bias.hex"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,

    // Input feature-map read port (active layer → buffer in top module)
    output wire [9:0]  in_addr,
    input  wire signed [15:0] in_data,

    // Output feature-map write port
    output reg  [9:0]  out_addr,
    output reg  signed [15:0] out_data,
    output reg         out_we
);

    // Derived constants
    localparam OUT_SIZE    = IN_SIZE - KERNEL + 1;
    localparam NUM_WEIGHTS = OUT_CH * IN_CH * KERNEL * KERNEL;
    localparam NUM_BIASES  = OUT_CH;

    // ---- Weight / Bias ROM ------------------------------------------------
    reg signed [15:0] weights [0:NUM_WEIGHTS-1];
    reg signed [15:0] biases  [0:NUM_BIASES-1];
    initial begin
        $readmemh(WEIGHT_FILE, weights);
        $readmemh(BIAS_FILE,   biases);
    end

    // ---- Loop counters (8-bit each, sufficient for all configs) -----------
    reg [7:0] oc;   // output channel
    reg [7:0] oy;   // output row
    reg [7:0] ox;   // output col
    reg [7:0] ic;   // input channel
    reg [7:0] ky;   // kernel row
    reg [7:0] kx;   // kernel col

    // ---- Accumulator (Q16.16 to hold sum-of-products) ---------------------
    reg signed [31:0] acc;

    // ---- FSM --------------------------------------------------------------
    localparam S_IDLE  = 3'd0,
               S_BIAS  = 3'd1,
               S_MAC   = 3'd2,
               S_WRITE = 3'd3,
               S_DONE  = 3'd4;
    reg [2:0] state;

    // ---- Combinational address into input buffer --------------------------
    // Layout: channel-major  addr = ic*IN_SIZE*IN_SIZE + (oy+ky)*IN_SIZE + (ox+kx)
    assign in_addr = ic * (IN_SIZE * IN_SIZE)
                   + (oy + ky) * IN_SIZE
                   + (ox + kx);

    // ---- Combinational weight lookup --------------------------------------
    wire [15:0] w_idx;
    assign w_idx = oc * (IN_CH * KERNEL * KERNEL)
                 + ic * (KERNEL * KERNEL)
                 + ky * KERNEL
                 + kx;

    wire signed [15:0] w_val = weights[w_idx];

    // ---- Multiply (Q8.8 × Q8.8 = Q16.16) ---------------------------------
    wire signed [31:0] mult = in_data * w_val;

    // ---- Q16.16 → Q8.8 with saturation & ReLU ----------------------------
    wire signed [15:0] acc_q88 = acc[23:8];
    wire overflow = (acc[31:24] != {8{acc[23]}});
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
            out_addr <= 10'd0;
            out_data <= 16'sd0;
        end else begin
            out_we <= 1'b0;   // default: no write

            case (state)
                // -----------------------------------------------------------
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        oc <= 0; oy <= 0; ox <= 0;
                        state <= S_BIAS;
                    end
                end

                // -----------------------------------------------------------
                S_BIAS: begin
                    // Load bias into accumulator, shifted to Q16.16
                    acc <= { {8{biases[oc][15]}}, biases[oc], 8'b0 };
                    ic  <= 0; ky <= 0; kx <= 0;
                    state <= S_MAC;
                end

                // -----------------------------------------------------------
                S_MAC: begin
                    // Accumulate one product per clock
                    acc <= acc + mult;

                    // Advance innermost → outermost: kx → ky → ic
                    if (kx == KERNEL - 1) begin
                        kx <= 0;
                        if (ky == KERNEL - 1) begin
                            ky <= 0;
                            if (ic == IN_CH - 1) begin
                                state <= S_WRITE;
                            end else begin
                                ic <= ic + 1;
                            end
                        end else begin
                            ky <= ky + 1;
                        end
                    end else begin
                        kx <= kx + 1;
                    end
                end

                // -----------------------------------------------------------
                S_WRITE: begin
                    // Output address: oc*OUT_SIZE*OUT_SIZE + oy*OUT_SIZE + ox
                    out_addr <= oc * (OUT_SIZE * OUT_SIZE) + oy * OUT_SIZE + ox;
                    out_data <= relu_out;
                    out_we   <= 1'b1;

                    // Advance output position: ox → oy → oc
                    if (ox == OUT_SIZE - 1) begin
                        ox <= 0;
                        if (oy == OUT_SIZE - 1) begin
                            oy <= 0;
                            if (oc == OUT_CH - 1) begin
                                state <= S_DONE;
                            end else begin
                                oc <= oc + 1;
                                state <= S_BIAS;
                            end
                        end else begin
                            oy <= oy + 1;
                            state <= S_BIAS;
                        end
                    end else begin
                        ox <= ox + 1;
                        state <= S_BIAS;
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
