// ============================================================================
// ReLU Activation (Combinational)
// f(x) = max(0, x) for signed fixed-point values
// ============================================================================
module relu #(
    parameter DATA_WIDTH = 16
)(
    input  wire signed [DATA_WIDTH-1:0] in,
    output wire signed [DATA_WIDTH-1:0] out
);
    // If MSB (sign bit) is 1 → negative → output 0; else pass through
    assign out = in[DATA_WIDTH-1] ? {DATA_WIDTH{1'b0}} : in;
endmodule
