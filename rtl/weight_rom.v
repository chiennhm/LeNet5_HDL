// ============================================================================
// Synchronous ROM for weight / bias storage.
// Inferred as Cyclone II M4K block RAM by Quartus.
// Initialized from a hex file at synthesis / simulation time.
// ============================================================================
module weight_rom #(
    parameter ADDR_W   = 15,
    parameter DATA_W   = 8,
    parameter DEPTH    = 32768,
    parameter MEM_FILE = ""
)(
    input  wire              clk,
    input  wire [ADDR_W-1:0] addr,
    output reg  signed [DATA_W-1:0] data
);

    (* ramstyle = "M4K" *)
    reg signed [DATA_W-1:0] mem [0:DEPTH-1];

    initial $readmemh(MEM_FILE, mem);

    always @(posedge clk)
        data <= mem[addr];

endmodule
