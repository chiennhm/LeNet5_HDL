// ============================================================================
// DE2 SRAM Controller — supports byte read AND byte write
// Replaces the read-only sram_read_bridge_de2.
// Write requests (from UART loader) have priority over read requests
// (from LeNet core), but in practice they never overlap.
// ============================================================================
module sram_controller_de2 (
    input  wire        clk,
    input  wire        rst_n,

    // Byte-read interface (from LeNet core)
    input  wire        rd_req,
    input  wire [23:0] rd_addr,
    output reg  signed [7:0] rd_data,
    output reg         rd_valid,

    // Byte-write interface (from UART protocol handler)
    input  wire        wr_req,
    input  wire [23:0] wr_addr,
    input  wire [7:0]  wr_data,
    output reg         wr_done,

    // DE2 physical SRAM pins
    output reg  [17:0] SRAM_ADDR,
    inout  wire [15:0] SRAM_DQ,
    output reg         SRAM_CE_N,
    output reg         SRAM_OE_N,
    output reg         SRAM_WE_N,
    output reg         SRAM_UB_N,
    output reg         SRAM_LB_N
);

    // ---- states -------------------------------------------------------------
    localparam S_IDLE       = 3'd0;
    localparam S_RD_WAIT    = 3'd1;
    localparam S_RD_CAPTURE = 3'd2;
    localparam S_WR_SETUP   = 3'd3;
    localparam S_WR_PULSE   = 3'd4;
    localparam S_WR_HOLD    = 3'd5;
    localparam S_WR_DONE    = 3'd6;

    reg [2:0]  state;
    reg        byte_sel;

    // Tri-state control for writes
    reg [15:0] sram_dq_out;
    reg        sram_dq_oe;

    assign SRAM_DQ = sram_dq_oe ? sram_dq_out : 16'hZZZZ;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            byte_sel    <= 1'b0;
            rd_data     <= 8'sd0;
            rd_valid    <= 1'b0;
            wr_done     <= 1'b0;
            SRAM_ADDR   <= 18'd0;
            SRAM_CE_N   <= 1'b1;
            SRAM_OE_N   <= 1'b1;
            SRAM_WE_N   <= 1'b1;
            SRAM_UB_N   <= 1'b1;
            SRAM_LB_N   <= 1'b1;
            sram_dq_out <= 16'd0;
            sram_dq_oe  <= 1'b0;
        end else begin
            rd_valid <= 1'b0;
            wr_done  <= 1'b0;

            case (state)
                S_IDLE: begin
                    SRAM_CE_N  <= 1'b1;
                    SRAM_OE_N  <= 1'b1;
                    SRAM_WE_N  <= 1'b1;
                    SRAM_UB_N  <= 1'b1;
                    SRAM_LB_N  <= 1'b1;
                    sram_dq_oe <= 1'b0;

                    if (wr_req) begin
                        // --- byte write (priority) -----------------------
                        SRAM_ADDR  <= wr_addr[18:1];
                        SRAM_CE_N  <= 1'b0;
                        SRAM_OE_N  <= 1'b1;
                        SRAM_WE_N  <= 1'b1; // Setup: WE_N stays HIGH
                        // Select the correct byte lane
                        SRAM_UB_N  <= ~wr_addr[0]; // odd  addr → upper
                        SRAM_LB_N  <=  wr_addr[0]; // even addr → lower
                        sram_dq_oe <= 1'b1;
                        if (wr_addr[0])
                            sram_dq_out <= {wr_data, 8'h00};
                        else
                            sram_dq_out <= {8'h00, wr_data};
                        state <= S_WR_SETUP;
                    end else if (rd_req) begin
                        // --- byte read --------------------------------
                        SRAM_ADDR  <= rd_addr[18:1];
                        byte_sel   <= rd_addr[0];
                        SRAM_CE_N  <= 1'b0;
                        SRAM_OE_N  <= 1'b0;
                        SRAM_WE_N  <= 1'b1;
                        SRAM_UB_N  <= 1'b0;   // enable both bytes for read
                        SRAM_LB_N  <= 1'b0;
                        sram_dq_oe <= 1'b0;
                        state      <= S_RD_WAIT;
                    end
                end

                // ---- read path -------------------------------------------
                S_RD_WAIT: begin
                    state <= S_RD_CAPTURE;     // 1-cycle SRAM access time
                end

                S_RD_CAPTURE: begin
                    rd_data   <= byte_sel ? SRAM_DQ[15:8] : SRAM_DQ[7:0];
                    rd_valid  <= 1'b1;
                    SRAM_CE_N <= 1'b1;
                    SRAM_OE_N <= 1'b1;
                    SRAM_UB_N <= 1'b1;
                    SRAM_LB_N <= 1'b1;
                    state     <= S_IDLE;
                end

                // ---- write path ------------------------------------------
                S_WR_SETUP: begin
                    SRAM_WE_N <= 1'b0; // Assert WE_N AFTER address/data setup
                    state <= S_WR_PULSE;
                end

                S_WR_PULSE: begin
                    SRAM_WE_N <= 1'b1; // De-assert WE_N. Data is written here!
                    state <= S_WR_HOLD;
                end

                S_WR_HOLD: begin
                    // Address and Data are still held stable for 1 cycle
                    // to satisfy hold time requirements after WE_N rises.
                    state <= S_WR_DONE;
                end

                S_WR_DONE: begin
                    SRAM_CE_N  <= 1'b1;
                    SRAM_UB_N  <= 1'b1;
                    SRAM_LB_N  <= 1'b1;
                    sram_dq_oe <= 1'b0;
                    wr_done    <= 1'b1;
                    state      <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
