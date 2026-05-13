// ============================================================================
// DE2 board wrapper for LeNet-5 core
// - Uses CLOCK_50 as system clock
// - Uses KEY[0] as active-low reset
// - Loads one 28x28 test image from hex into core input buffer at boot
// - Reads weights from external SRAM through sram_controller_de2
// - Press KEY[1] to start inference
// - UART (9600-8N1) for PC communication: weight loading, inference, status
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
    output wire        SRAM_CE_N,

    // UART pins (directly exposed on DE2 header)
    input  wire        UART_RXD,
    output wire        UART_TXD
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

    // SRAM write channel from UART protocol handler.
    wire        uart_sram_wr_req;
    wire [23:0] uart_sram_wr_addr;
    wire [7:0]  uart_sram_wr_data;
    wire        uart_sram_wr_done;

    // UART core-control signals from protocol handler.
    wire        uart_core_start;

    // UART image write channel from protocol handler.
    wire        uart_img_wr_en;
    wire [9:0]  uart_img_wr_addr;
    wire [7:0]  uart_img_wr_data;

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
            end else if (uart_img_wr_en) begin
                pixel_addr <= uart_img_wr_addr;
                pixel_data <= uart_img_wr_data;
                pixel_we   <= 1'b1;
            end else if (start_btn_pulse || uart_core_start) begin
                start <= 1'b1;
            end
        end
    end

    // ================================================================
    //  LeNet-5 Core
    // ================================================================
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

    // ================================================================
    //  Unified SRAM Controller (read + write)
    // ================================================================
    sram_controller_de2 u_sram_ctrl (
        .clk      (clk),
        .rst_n    (rst_n),
        // Read port (LeNet core)
        .rd_req   (core_sram_rd_req),
        .rd_addr  (core_sram_rd_addr),
        .rd_data  (core_sram_rd_data),
        .rd_valid (core_sram_rd_valid),
        // Write port (UART loader)
        .wr_req   (uart_sram_wr_req),
        .wr_addr  (uart_sram_wr_addr),
        .wr_data  (uart_sram_wr_data),
        .wr_done  (uart_sram_wr_done),
        // Physical SRAM pins
        .SRAM_ADDR(SRAM_ADDR),
        .SRAM_DQ  (SRAM_DQ),
        .SRAM_CE_N(SRAM_CE_N),
        .SRAM_OE_N(SRAM_OE_N),
        .SRAM_WE_N(SRAM_WE_N),
        .SRAM_UB_N(SRAM_UB_N),
        .SRAM_LB_N(SRAM_LB_N)
    );

    // ================================================================
    //  UART RX / TX
    // ================================================================
    wire [7:0] uart_rx_data;
    wire       uart_rx_valid;
    wire [7:0] uart_tx_data;
    wire       uart_tx_start;
    wire       uart_tx_busy;

    uart_rx #(
        .CLK_FREQ  (50_000_000),
        .BAUD_RATE (115200)
    ) u_uart_rx (
        .clk      (clk),
        .rst_n    (rst_n),
        .rx_in    (UART_RXD),
        .rx_data  (uart_rx_data),
        .rx_valid (uart_rx_valid)
    );

    uart_tx #(
        .CLK_FREQ  (50_000_000),
        .BAUD_RATE (115200)
    ) u_uart_tx (
        .clk      (clk),
        .rst_n    (rst_n),
        .tx_data  (uart_tx_data),
        .tx_start (uart_tx_start),
        .tx_out   (UART_TXD),
        .tx_busy  (uart_tx_busy)
    );

    // ================================================================
    //  UART Protocol Handler
    // ================================================================
    uart_protocol u_uart_proto (
        .clk          (clk),
        .rst_n        (rst_n),
        // UART byte interface
        .rx_data      (uart_rx_data),
        .rx_valid     (uart_rx_valid),
        .tx_data      (uart_tx_data),
        .tx_start     (uart_tx_start),
        .tx_busy      (uart_tx_busy),
        // SRAM write
        .sram_wr_req  (uart_sram_wr_req),
        .sram_wr_addr (uart_sram_wr_addr),
        .sram_wr_data (uart_sram_wr_data),
        .sram_wr_done (uart_sram_wr_done),
        // Core control
        .core_start   (uart_core_start),
        .core_done    (done),
        .core_digit   (digit_out),
        // Image buffer write
        .img_wr_en    (uart_img_wr_en),
        .img_wr_addr  (uart_img_wr_addr),
        .img_wr_data  (uart_img_wr_data)
    );

    // ================================================================
    //  7-Segment Decoder
    // ================================================================
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
