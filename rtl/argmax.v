// ============================================================================
// Argmax Module
// Sequentially reads NUM_CLASSES signed values and outputs the index of the
// largest value. Result is available when 'done' is asserted.
// ============================================================================
module argmax #(
    parameter NUM_CLASSES = 10,
    parameter DATA_WIDTH  = 16
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    output wire [9:0]              in_addr,   // address to read input buffer
    input  wire signed [DATA_WIDTH-1:0] in_data,
    output reg                     done,
    output reg  [3:0]              class_out
);

    reg [3:0] cnt;
    reg signed [DATA_WIDTH-1:0] max_val;
    reg [3:0] max_idx;

    localparam S_IDLE = 2'd0,
               S_RUN  = 2'd1,
               S_OUT  = 2'd2;
    reg [1:0] state;

    // Address is the current counter (combinational read)
    assign in_addr = {6'b0, cnt};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            done      <= 1'b0;
            cnt       <= 4'd0;
            max_val   <= {1'b1, {(DATA_WIDTH-1){1'b0}}}; // most negative
            max_idx   <= 4'd0;
            class_out <= 4'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        cnt     <= 4'd0;
                        max_val <= {1'b1, {(DATA_WIDTH-1){1'b0}}};
                        max_idx <= 4'd0;
                        state   <= S_RUN;
                    end
                end

                S_RUN: begin
                    // Compare current input with running max
                    if (in_data > max_val) begin
                        max_val <= in_data;
                        max_idx <= cnt;
                    end
                    if (cnt == NUM_CLASSES - 1) begin
                        state <= S_OUT;
                    end else begin
                        cnt <= cnt + 4'd1;
                    end
                end

                S_OUT: begin
                    class_out <= max_idx;
                    done      <= 1'b1;
                    state     <= S_IDLE;
                end
            endcase
        end
    end
endmodule
