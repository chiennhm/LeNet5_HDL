// ============================================================================
// DE2 Asynchronous SRAM read bridge
// Converts byte-address read requests into DE2 SRAM pin transactions.
// One read request is served at a time.
// ============================================================================
module sram_read_bridge_de2 (
    input  wire        clk,
    input  wire        rst_n,

    // Simple byte-read request interface from accelerator core
    input  wire        rd_req,
    input  wire [23:0] rd_addr,
    output reg  signed [7:0] rd_data,
    output reg         rd_valid,

    // DE2 SRAM device pins
    output reg  [17:0] SRAM_ADDR,
    inout  wire [15:0] SRAM_DQ,
    output reg         SRAM_CE_N,
    output reg         SRAM_OE_N,
    output reg         SRAM_WE_N,
    output reg         SRAM_UB_N,
    output reg         SRAM_LB_N
);

    localparam S_IDLE    = 2'd0;
    localparam S_WAIT    = 2'd1;
    localparam S_CAPTURE = 2'd2;

    reg [1:0] state;
    reg       byte_sel;

    // Read-only bridge: never drives SRAM data bus.
    assign SRAM_DQ = 16'hZZZZ;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            byte_sel   <= 1'b0;
            rd_data    <= 8'sd0;
            rd_valid   <= 1'b0;
            SRAM_ADDR  <= 18'd0;
            SRAM_CE_N  <= 1'b1;
            SRAM_OE_N  <= 1'b1;
            SRAM_WE_N  <= 1'b1;
            SRAM_UB_N  <= 1'b1;
            SRAM_LB_N  <= 1'b1;
        end else begin
            rd_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    SRAM_CE_N <= 1'b1;
                    SRAM_OE_N <= 1'b1;
                    SRAM_WE_N <= 1'b1;
                    SRAM_UB_N <= 1'b1;
                    SRAM_LB_N <= 1'b1;

                    if (rd_req) begin
                        // Byte-address to word-address conversion.
                        SRAM_ADDR <= rd_addr[18:1];
                        byte_sel  <= rd_addr[0];

                        SRAM_CE_N <= 1'b0;
                        SRAM_OE_N <= 1'b0;
                        SRAM_WE_N <= 1'b1;
                        SRAM_UB_N <= ~rd_addr[0];
                        SRAM_LB_N <=  rd_addr[0];
                        state     <= S_WAIT;
                    end
                end

                S_WAIT: begin
                    // Wait one cycle for asynchronous SRAM access time.
                    state <= S_CAPTURE;
                end

                S_CAPTURE: begin
                    rd_data  <= byte_sel ? SRAM_DQ[15:8] : SRAM_DQ[7:0];
                    rd_valid <= 1'b1;

                    SRAM_CE_N <= 1'b1;
                    SRAM_OE_N <= 1'b1;
                    SRAM_WE_N <= 1'b1;
                    SRAM_UB_N <= 1'b1;
                    SRAM_LB_N <= 1'b1;
                    state     <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
