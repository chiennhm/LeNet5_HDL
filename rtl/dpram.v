// ============================================================================
// Simple Dual-Port RAM  (1 read port, 1 write port)
// Inferred as Cyclone II M4K block RAM by Quartus.
// ============================================================================
module dpram #(
    parameter ADDR_W = 12,
    parameter DATA_W = 16,
    parameter DEPTH  = 3456
)(
    input  wire              clk,
    // Write port
    input  wire              we,
    input  wire [ADDR_W-1:0] waddr,
    input  wire signed [DATA_W-1:0] wdata,
    // Read port
    input  wire [ADDR_W-1:0] raddr,
    output reg  signed [DATA_W-1:0] rdata
);

    (* ramstyle = "no_rw_check, M4K" *)
    reg signed [DATA_W-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rdata <= mem[raddr];
    end

endmodule
