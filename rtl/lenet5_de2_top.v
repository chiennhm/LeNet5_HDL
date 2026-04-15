// ============================================================================
// DE2 board wrapper for LeNet-5 core
// - Uses CLOCK_50 as system clock
// - Uses KEY[0] as active-low reset
// - Loads one 28x28 test image from hex into core input buffer at boot
// - Reads weights from external SRAM through sram_read_bridge_de2
// - Press KEY[1] to start inference
// ============================================================================
module lenet5_de2_top (
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    input  wire [17:0] SW,
    output wire [17:0] LEDR,
    output wire [8:0]  LEDG,
    output wire [6:0]  HEX0,
    output wire [6:0]  HEX1,
    output wire [6:0]  HEX2,
    output wire [6:0]  HEX3,
    output wire [6:0]  HEX4,
    output wire [6:0]  HEX5,
    output wire [6:0]  HEX6,
    output wire [6:0]  HEX7,

    output wire [17:0] SRAM_ADDR,
    inout  wire [15:0] SRAM_DQ,
    output wire        SRAM_WE_N,
    output wire        SRAM_OE_N,
    output wire        SRAM_UB_N,
    output wire        SRAM_LB_N,
    output wire        SRAM_CE_N
);

    localparam IMG_SIZE = 10'd784;

    wire clk   = CLOCK_50;
    wire rst_n = KEY[0];

    // KEY on DE2 is active-low. Generate one-cycle start pulse on press.
    reg key1_d;
    wire start_btn_pulse = key1_d & ~KEY[1];

    // Image loading channel toward LeNet core.
    reg  [7:0] pixel_data;
    reg  [9:0] pixel_addr;
    reg        pixel_we;
    reg        start;

    wire       done;
    wire [3:0] digit_out;

    // SRAM read request channel between core and board bridge.
    wire        core_sram_rd_req;
    wire [23:0] core_sram_rd_addr;
    wire signed [7:0] core_sram_rd_data;
    wire        core_sram_rd_valid;

    // Boot-time image loader.
    reg        loading_img;
    reg [9:0]  load_idx;
    reg [7:0]  image_mem [0:783];

    initial begin
        $readmemh("mem/test_image.hex", image_mem);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            key1_d      <= 1'b1;
            loading_img <= 1'b1;
            load_idx    <= 10'd0;
            pixel_data  <= 8'd0;
            pixel_addr  <= 10'd0;
            pixel_we    <= 1'b0;
            start       <= 1'b0;
        end else begin
            key1_d   <= KEY[1];
            pixel_we <= 1'b0;
            start    <= 1'b0;

            if (loading_img) begin
                pixel_addr <= load_idx;
                pixel_data <= image_mem[load_idx];
                pixel_we   <= 1'b1;

                if (load_idx == IMG_SIZE - 1) begin
                    loading_img <= 1'b0;
                end else begin
                    load_idx <= load_idx + 10'd1;
                end
            end else if (start_btn_pulse) begin
                start <= 1'b1;
            end
        end
    end

    // Core instance
    lenet5_top u_core (
        .clk           (clk),
        .rst_n         (rst_n),
        .pixel_data    (pixel_data),
        .pixel_addr    (pixel_addr),
        .pixel_we      (pixel_we),
        .start         (start),
        .done          (done),
        .digit_out     (digit_out),
        .sram_rd_req   (core_sram_rd_req),
        .sram_rd_addr  (core_sram_rd_addr),
        .sram_rd_data  (core_sram_rd_data),
        .sram_rd_valid (core_sram_rd_valid)
    );

    // Physical SRAM read bridge
    sram_read_bridge_de2 u_sram_bridge (
        .clk      (clk),
        .rst_n    (rst_n),
        .rd_req   (core_sram_rd_req),
        .rd_addr  (core_sram_rd_addr),
        .rd_data  (core_sram_rd_data),
        .rd_valid (core_sram_rd_valid),
        .SRAM_ADDR(SRAM_ADDR),
        .SRAM_DQ  (SRAM_DQ),
        .SRAM_CE_N(SRAM_CE_N),
        .SRAM_OE_N(SRAM_OE_N),
        .SRAM_WE_N(SRAM_WE_N),
        .SRAM_UB_N(SRAM_UB_N),
        .SRAM_LB_N(SRAM_LB_N)
    );

    function [6:0] seg7;
        input [3:0] val;
        begin
            case (val)
                4'h0: seg7 = 7'b1000000;
                4'h1: seg7 = 7'b1111001;
                4'h2: seg7 = 7'b0100100;
                4'h3: seg7 = 7'b0110000;
                4'h4: seg7 = 7'b0011001;
                4'h5: seg7 = 7'b0010010;
                4'h6: seg7 = 7'b0000010;
                4'h7: seg7 = 7'b1111000;
                4'h8: seg7 = 7'b0000000;
                4'h9: seg7 = 7'b0010000;
                4'hA: seg7 = 7'b0001000;
                4'hB: seg7 = 7'b0000011;
                4'hC: seg7 = 7'b1000110;
                4'hD: seg7 = 7'b0100001;
                4'hE: seg7 = 7'b0000110;
                4'hF: seg7 = 7'b0001110;
                default: seg7 = 7'b1111111;
            endcase
        end
    endfunction

    assign LEDR[3:0]   = digit_out;
    assign LEDR[16:4]  = 13'd0;
    assign LEDR[17]    = done;

    assign LEDG[0]     = done;
    assign LEDG[1]     = ~loading_img;
    assign LEDG[8:2]   = 7'd0;

    assign HEX0 = seg7(digit_out);
    assign HEX1 = 7'b1111111;
    assign HEX2 = 7'b1111111;
    assign HEX3 = 7'b1111111;
    assign HEX4 = 7'b1111111;
    assign HEX5 = 7'b1111111;
    assign HEX6 = 7'b1111111;
    assign HEX7 = 7'b1111111;

    // SW is currently unused but kept for future control features.
    wire [17:0] sw_unused = SW;

endmodule
