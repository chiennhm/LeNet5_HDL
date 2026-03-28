// ============================================================================
// Testbench for LeNet-5 Top Module
// 1. Loads a 28×28 test image from test_image.hex
// 2. Writes pixels into the design
// 3. Asserts start, waits for done
// 4. Prints predicted digit
// ============================================================================
`timescale 1ns / 1ps

module tb_lenet5;

    // ---- Clock & reset ----------------------------------------------------
    reg clk, rst_n;
    localparam CLK_PERIOD = 10;  // 100 MHz

    always #(CLK_PERIOD/2) clk = ~clk;

    // ---- DUT signals -------------------------------------------------------
    reg  [7:0] pixel_data;
    reg  [9:0] pixel_addr;
    reg        pixel_we;
    reg        start;
    wire       done;
    wire [3:0] digit_out;

    // ---- DUT ---------------------------------------------------------------
    lenet5_top u_dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .pixel_data (pixel_data),
        .pixel_addr (pixel_addr),
        .pixel_we   (pixel_we),
        .start      (start),
        .done       (done),
        .digit_out  (digit_out)
    );

    // ---- Test image memory -------------------------------------------------
    reg [7:0] test_img [0:783];
    initial $readmemh("test_image.hex", test_img);

    // ---- Cycle counter for performance measurement -------------------------
    integer cycle_cnt;
    always @(posedge clk) begin
        if (start) cycle_cnt <= 0;
        else if (!done) cycle_cnt <= cycle_cnt + 1;
    end

    // ---- Waveform dump -----------------------------------------------------
    initial begin
        $dumpfile("lenet5_wave.vcd");
        $dumpvars(0, tb_lenet5);
    end

    // ---- Main test sequence ------------------------------------------------
    integer i;
    initial begin
        clk       = 0;
        rst_n     = 0;
        pixel_we  = 0;
        pixel_data = 0;
        pixel_addr = 0;
        start     = 0;
        cycle_cnt = 0;

        // Reset
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        // ----- Load image into DUT -----------------------------------------
        $display("========================================");
        $display(" LeNet-5 MNIST 28x28 Inference Test");
        $display("========================================");
        $display("[%0t] Loading test image (784 pixels)...", $time);

        for (i = 0; i < 784; i = i + 1) begin
            @(posedge clk);
            pixel_addr <= i;
            pixel_data <= test_img[i];
            pixel_we   <= 1'b1;
        end
        @(posedge clk);
        pixel_we <= 1'b0;

        // ----- Start inference ---------------------------------------------
        $display("[%0t] Image loaded. Starting inference...", $time);
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        // ----- Wait for done -----------------------------------------------
        wait (done == 1'b1);
        @(posedge clk);  // one more cycle to settle

        $display("[%0t] Inference complete!", $time);
        $display("  Predicted digit : %0d", digit_out);
        $display("  Inference cycles: %0d", cycle_cnt);
        $display("========================================");

        #(CLK_PERIOD * 10);
        $finish;
    end

    // ---- Timeout watchdog -------------------------------------------------
    initial begin
        #(CLK_PERIOD * 5000000);  // 5M cycles max (larger network)
        $display("ERROR: Simulation timed out!");
        $finish;
    end

endmodule
