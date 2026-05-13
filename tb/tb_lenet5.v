// ============================================================================
// Testbench for LeNet-5 Top Module with external SRAM weights
// 1. Host writes all model weights into a simulated SRAM
// 2. Host writes a 28x28 test image into DUT input buffer
// 3. DUT runs inference while fetching weights from SRAM
// 4. Testbench reports predicted digit and cycle count
// ============================================================================
`timescale 1ns / 1ps

`ifndef IMG_HEX_FILE
`define IMG_HEX_FILE "mem/test_img_61_label_8.hex"
`endif

module sram_weight_model #(
    parameter ADDR_W     = 24,
    parameter DATA_W     = 8,
    parameter DEPTH      = 65536,
    parameter RD_LATENCY = 1
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     rd_req,
    input  wire [ADDR_W-1:0]        rd_addr,
    output reg  signed [DATA_W-1:0] rd_data,
    output reg                      rd_valid,
    input  wire                     wr_en,
    input  wire [ADDR_W-1:0]        wr_addr,
    input  wire signed [DATA_W-1:0] wr_data
);
    reg signed [DATA_W-1:0] mem [0:DEPTH-1];
    reg [ADDR_W-1:0] addr_pipe [0:RD_LATENCY-1];
    reg              valid_pipe[0:RD_LATENCY-1];
    integer p;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data  <= {DATA_W{1'b0}};
            rd_valid <= 1'b0;
            for (p = 0; p < RD_LATENCY; p = p + 1) begin
                addr_pipe[p]  <= {ADDR_W{1'b0}};
                valid_pipe[p] <= 1'b0;
            end
        end else begin
            if (wr_en) begin
                mem[wr_addr] <= wr_data;
            end

            addr_pipe[0]  <= rd_addr;
            valid_pipe[0] <= rd_req;
            for (p = 1; p < RD_LATENCY; p = p + 1) begin
                addr_pipe[p]  <= addr_pipe[p-1];
                valid_pipe[p] <= valid_pipe[p-1];
            end

            rd_valid <= valid_pipe[RD_LATENCY-1];
            if (valid_pipe[RD_LATENCY-1]) begin
                rd_data <= mem[addr_pipe[RD_LATENCY-1]];
            end
        end
    end
endmodule

module tb_lenet5;

    // ---- Clock & reset ----------------------------------------------------
    reg clk;
    reg rst_n;
    localparam CLK_PERIOD = 10;  // 100 MHz
    always #(CLK_PERIOD/2) clk = ~clk;

    // ---- Shared SRAM address map ------------------------------------------
    localparam [23:0] WBASE_C1  = 24'd0;
    localparam [23:0] WBASE_C3  = 24'd150;
    localparam [23:0] WBASE_C5  = 24'd2550;
    localparam [23:0] WBASE_FC1 = 24'd33270;
    localparam [23:0] WBASE_FC2 = 24'd43350;

    // ---- DUT image/control signals ----------------------------------------
    reg  [7:0] pixel_data;
    reg  [9:0] pixel_addr;
    reg        pixel_we;
    reg        start;
    wire       done;
    wire [3:0] digit_out;

    // ---- DUT <-> SRAM read channel ----------------------------------------
    wire        sram_rd_req;
    wire [23:0] sram_rd_addr;
    wire signed [7:0] sram_rd_data;
    wire        sram_rd_valid;

    // ---- Host write channel to SRAM model ---------------------------------
    reg         host_wr_en;
    reg  [23:0] host_wr_addr;
    reg  signed [7:0] host_wr_data;

    // ---- DUT ----------------------------------------------------------------
    lenet5_top u_dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .pixel_data    (pixel_data),
        .pixel_addr    (pixel_addr),
        .pixel_we      (pixel_we),
        .start         (start),
        .done          (done),
        .digit_out     (digit_out),
        .sram_rd_req   (sram_rd_req),
        .sram_rd_addr  (sram_rd_addr),
        .sram_rd_data  (sram_rd_data),
        .sram_rd_valid (sram_rd_valid)
    );

    // ---- Simulated SRAM ----------------------------------------------------
    sram_weight_model #(
        .ADDR_W(24), .DATA_W(8), .DEPTH(65536), .RD_LATENCY(1)
    ) u_sram (
        .clk     (clk),
        .rst_n   (rst_n),
        .rd_req  (sram_rd_req),
        .rd_addr (sram_rd_addr),
        .rd_data (sram_rd_data),
        .rd_valid(sram_rd_valid),
        .wr_en   (host_wr_en),
        .wr_addr (host_wr_addr),
        .wr_data (host_wr_data)
    );

    // ---- Weight/image files loaded by host model --------------------------
    reg signed [7:0] conv1_w [0:149];
    reg signed [7:0] conv3_w [0:2399];
    reg signed [7:0] c5_w    [0:30719];
    reg signed [7:0] fc1_w   [0:10079];
    reg signed [7:0] fc2_w   [0:839];
    reg        [7:0] test_img[0:783];
    integer k;
    integer img_nonzero_cnt;
    reg [7:0] img_max;

    initial begin
        $readmemh("mem/conv1_weights.hex", conv1_w);
        $readmemh("mem/conv3_weights.hex", conv3_w);
        $readmemh("mem/c5_weights.hex",    c5_w);
        $readmemh("mem/fc1_weights.hex",   fc1_w);
        $readmemh("mem/fc2_weights.hex",   fc2_w);
        $readmemh(`IMG_HEX_FILE,            test_img);

        // Catch missing/unreadable image file early.
        if (^test_img[0] === 1'bx) begin
            $display("ERROR: Failed to load image file: %s", `IMG_HEX_FILE);
            $finish;
        end

        img_nonzero_cnt = 0;
        img_max = 8'd0;
        for (k = 0; k < 784; k = k + 1) begin
            if (test_img[k] != 8'd0) begin
                img_nonzero_cnt = img_nonzero_cnt + 1;
            end
            if (test_img[k] > img_max) begin
                img_max = test_img[k];
            end
        end
        $display("Image stats: nonzero=%0d/784, max=%0d", img_nonzero_cnt, img_max);
    end

    // ---- Cycle counter (inference only) -----------------------------------
    integer cycle_cnt;
    reg inference_active;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_cnt <= 0;
            inference_active <= 1'b0;
        end else begin
            if (start) begin
                inference_active <= 1'b1;
                cycle_cnt <= 0;
            end else if (done) begin
                inference_active <= 1'b0;
            end else if (inference_active) begin
                cycle_cnt <= cycle_cnt + 1;
            end
        end
    end

    // ---- Waveform dump -----------------------------------------------------
    initial begin
        $dumpfile("lenet5_wave.vcd");
        $dumpvars(0, tb_lenet5);
    end

    // ---- Host-side SRAM single-byte write task ----------------------------
    task automatic sram_host_write;
        input [23:0] addr;
        input signed [7:0] data;
        begin
            host_wr_addr = addr;
            host_wr_data = data;
            host_wr_en   = 1'b1;
            @(posedge clk);
            host_wr_en   = 1'b0;
        end
    endtask

    // ---- Main test sequence ------------------------------------------------
    integer i;
    integer img_err_cnt;
    initial begin
        clk          = 1'b0;
        rst_n        = 1'b0;
        pixel_we     = 1'b0;
        pixel_data   = 8'd0;
        pixel_addr   = 10'd0;
        start        = 1'b0;
        host_wr_en   = 1'b0;
        host_wr_addr = 24'd0;
        host_wr_data = 8'sd0;

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        #(CLK_PERIOD * 2);

        $display("========================================");
        $display(" LeNet-5 SRAM Weight Loading + Inference");
        $display("========================================");
        $display("[%0t] Host is loading model weights into SRAM...", $time);

        for (i = 0; i < 150; i = i + 1)
            sram_host_write(WBASE_C1 + i, conv1_w[i]);

        for (i = 0; i < 2400; i = i + 1)
            sram_host_write(WBASE_C3 + i, conv3_w[i]);

        for (i = 0; i < 30720; i = i + 1)
            sram_host_write(WBASE_C5 + i, c5_w[i]);

        for (i = 0; i < 10080; i = i + 1)
            sram_host_write(WBASE_FC1 + i, fc1_w[i]);

        for (i = 0; i < 840; i = i + 1)
            sram_host_write(WBASE_FC2 + i, fc2_w[i]);

        $display("[%0t] SRAM weight loading complete.", $time);
        $display("[%0t] Loading 28x28 test image into DUT...", $time);

        for (i = 0; i < 784; i = i + 1) begin
            // Drive write signals before the active edge to avoid race
            // with synchronous RAM sampling in the DUT.
            pixel_addr = i[9:0];
            pixel_data = test_img[i];
            pixel_we   = 1'b1;
            @(posedge clk);
        end

        pixel_we   = 1'b0;
        pixel_addr = 10'd0;
        pixel_data = 8'd0;
        @(posedge clk);

        // Verify that all 784 bytes were written into DUT input buffer.
        img_err_cnt = 0;
        for (i = 0; i < 784; i = i + 1) begin
            if (u_dut.u_buf_input.mem[i][7:0] !== test_img[i]) begin
                img_err_cnt = img_err_cnt + 1;
            end
        end
        if (img_err_cnt != 0) begin
            $display("ERROR: Image load mismatch count = %0d", img_err_cnt);
            $finish;
        end
        $display("[%0t] Starting inference...", $time);
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        wait (done == 1'b1);
        @(posedge clk);

        $display("[%0t] Inference complete!", $time);
        $display("  Predicted digit : %0d", digit_out);
        $display("  Inference cycles: %0d", cycle_cnt);
        $display("========================================");

        #(CLK_PERIOD * 10);
        $finish;
    end

    // ---- Timeout watchdog --------------------------------------------------
    initial begin
        #(CLK_PERIOD * 15000000);
        $display("ERROR: Simulation timed out!");
        $finish;
    end

endmodule
