// ============================================================================
// Average-Pooling Layer Engine (2×2, stride 2)
// Reads input feature map via address/data port, writes pooled output.
// Computes mean of each 2×2 window: sum / 4 = sum >>> 2
// Fixed-point Q8.8 (16-bit signed).
// ============================================================================
module avgpool_layer #(
    parameter IN_SIZE  = 24,   // input spatial dimension (square)
    parameter CHANNELS = 6    // number of channels
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,

    // Input read port
    output wire [11:0] in_addr,
    input  wire signed [15:0] in_data,

    // Output write port
    output reg  [11:0] out_addr,
    output reg  signed [15:0] out_data,
    output reg         out_we
);

    localparam OUT_SIZE = IN_SIZE / 2;
    localparam POOL     = 2;

    // ---- Loop counters ----------------------------------------------------
    reg [7:0] ch;   // channel
    reg [7:0] oy;   // output row
    reg [7:0] ox;   // output col
    reg [1:0] ky;   // pool window row (0-1)
    reg [1:0] kx;   // pool window col (0-1)

    // Accumulator for sum (need extra bits: 16-bit * 4 → 18 bits sufficient)
    reg signed [17:0] sum;

    // ---- FSM --------------------------------------------------------------
    localparam S_IDLE  = 3'd0,
               S_INIT  = 3'd1,   // reset sum for new window
               S_ACC   = 3'd2,   // read & accumulate
               S_WRITE = 3'd3,
               S_DONE  = 3'd4;
    reg [2:0] state;

    // ---- Combinational read address ---------------------------------------
    // Layout: ch * IN_SIZE * IN_SIZE + (oy*2+ky) * IN_SIZE + (ox*2+kx)
    assign in_addr = ch * (IN_SIZE * IN_SIZE)
                   + (oy * POOL + ky) * IN_SIZE
                   + (ox * POOL + kx);

    // ---- Average: sum >>> 2 (arithmetic right shift for signed) -----------
    wire signed [15:0] avg_val = sum[17:2]; // sum / 4

    // ---- Main FSM ---------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            done    <= 1'b0;
            out_we  <= 1'b0;
            ch <= 0; oy <= 0; ox <= 0; ky <= 0; kx <= 0;
            sum     <= 18'sd0;
            out_addr <= 0;
            out_data <= 0;
        end else begin
            out_we <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        ch <= 0; oy <= 0; ox <= 0;
                        state <= S_INIT;
                    end
                end

                S_INIT: begin
                    sum <= 18'sd0;
                    ky <= 0; kx <= 0;
                    state <= S_ACC;
                end

                S_ACC: begin
                    // Accumulate current element
                    sum <= sum + {{2{in_data[15]}}, in_data};

                    // Advance within 2×2 window
                    if (kx == POOL - 1) begin
                        kx <= 0;
                        if (ky == POOL - 1) begin
                            state <= S_WRITE;
                        end else begin
                            ky <= ky + 1;
                        end
                    end else begin
                        kx <= kx + 1;
                    end
                end

                S_WRITE: begin
                    out_addr <= ch * (OUT_SIZE * OUT_SIZE) + oy * OUT_SIZE + ox;
                    out_data <= avg_val;
                    out_we   <= 1'b1;

                    // Advance output position
                    if (ox == OUT_SIZE - 1) begin
                        ox <= 0;
                        if (oy == OUT_SIZE - 1) begin
                            oy <= 0;
                            if (ch == CHANNELS - 1) begin
                                state <= S_DONE;
                            end else begin
                                ch <= ch + 1;
                                state <= S_INIT;
                            end
                        end else begin
                            oy <= oy + 1;
                            state <= S_INIT;
                        end
                    end else begin
                        ox <= ox + 1;
                        state <= S_INIT;
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
