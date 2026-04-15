// ============================================================================
// LeNet-5 Top-Level Module — Resource-Optimized for Cyclone II (DE2 Kit)
//
// Architecture (unchanged from Python model.py):
//   Conv2d(1→6, k=5) + ReLU → AvgPool(2) →
//   Conv2d(6→16, k=5) + ReLU → AvgPool(2) →
//   Conv2d(16→120, k=4) + ReLU → flatten →
//   Linear(120→84) + ReLU → Linear(84→10) → Argmax
//
// Resource optimisations:
//   ● Feature-map buffers → M4K dual-port RAM (dpram)
//   ● Weights in external SRAM, streamed by a shared read port
//   ● Biases remain as small internal register arrays per layer
//   ● No runtime multipliers for address calculation
//   ● MAC engines stall on weight-valid, tolerant to SRAM read latency
//
// Note: top-level now exposes an SRAM weight read interface.
// ============================================================================
module lenet5_top (
    input  wire        clk,
    input  wire        rst_n,

    // ---- Image load interface -------------------------------------------
    input  wire [7:0]  pixel_data,    // 8-bit grayscale pixel
    input  wire [9:0]  pixel_addr,    // address 0..783
    input  wire        pixel_we,      // write-enable

    // ---- Control & result -----------------------------------------------
    input  wire        start,
    output reg         done,
    output wire [3:0]  digit_out,

    // ---- External SRAM weight read interface ----------------------------
    output wire        sram_rd_req,
    output wire [23:0] sram_rd_addr,
    input  wire signed [7:0] sram_rd_data,
    input  wire        sram_rd_valid
);

    // =====================================================================
    //  FSM
    // =====================================================================
    localparam ST_IDLE      = 4'd0,
               ST_RUN_C1    = 4'd1,
               ST_RUN_S2    = 4'd2,
               ST_RUN_C3    = 4'd3,
               ST_RUN_S4    = 4'd4,
               ST_RUN_C5    = 4'd5,
               ST_RUN_FC1   = 4'd6,
               ST_RUN_FC2   = 4'd7,
               ST_RUN_AM    = 4'd8,
               ST_DONE      = 4'd9;
    reg [3:0] fsm;

    // =====================================================================
    //  LAYER START / DONE SIGNALS
    // =====================================================================
    reg  c1_start, s2_start, c3_start, s4_start, c5_start;
    reg  fc1_start, fc2_start, am_start;
    wire c1_done,  s2_done,  c3_done,  s4_done,  c5_done;
    wire fc1_done, fc2_done, am_done;

    // =====================================================================
    //  LAYER ↔ BUFFER ADDRESS / DATA WIRES
    // =====================================================================

    // -- C1 (reads buf_input, writes buf_a) --
    wire [11:0]        c1_rd_addr;
    wire [11:0]        c1_wr_addr;
    wire signed [15:0] c1_wr_data;
    wire               c1_wr_en;
    wire               c1_w_req;
    wire [14:0]        c1_w_addr;
    wire signed [7:0]  c1_w_data;
    wire               c1_w_valid;

    // -- S2 (reads buf_a, writes buf_b) --
    wire [11:0]        s2_rd_addr;
    wire [11:0]        s2_wr_addr;
    wire signed [15:0] s2_wr_data;
    wire               s2_wr_en;

    // -- C3 (reads buf_b, writes buf_a) --
    wire [11:0]        c3_rd_addr;
    wire [11:0]        c3_wr_addr;
    wire signed [15:0] c3_wr_data;
    wire               c3_wr_en;
    wire               c3_w_req;
    wire [14:0]        c3_w_addr;
    wire signed [7:0]  c3_w_data;
    wire               c3_w_valid;

    // -- S4 (reads buf_a, writes buf_b) --
    wire [11:0]        s4_rd_addr;
    wire [11:0]        s4_wr_addr;
    wire signed [15:0] s4_wr_data;
    wire               s4_wr_en;

    // -- C5 (reads buf_b, writes buf_a) --
    wire [11:0]        c5_rd_addr;
    wire [11:0]        c5_wr_addr;
    wire signed [15:0] c5_wr_data;
    wire               c5_wr_en;
    wire               c5_w_req;
    wire [14:0]        c5_w_addr;
    wire signed [7:0]  c5_w_data;
    wire               c5_w_valid;

    // -- FC1 (reads buf_a, writes buf_b) --
    wire [11:0]        fc1_rd_addr;
    wire [11:0]        fc1_wr_addr;
    wire signed [15:0] fc1_wr_data;
    wire               fc1_wr_en;
    wire               fc1_w_req;
    wire [14:0]        fc1_w_addr;
    wire signed [7:0]  fc1_w_data;
    wire               fc1_w_valid;

    // -- FC2 (reads buf_b, writes buf_a) --
    wire [11:0]        fc2_rd_addr;
    wire [11:0]        fc2_wr_addr;
    wire signed [15:0] fc2_wr_data;
    wire               fc2_wr_en;
    wire               fc2_w_req;
    wire [14:0]        fc2_w_addr;
    wire signed [7:0]  fc2_w_data;
    wire               fc2_w_valid;

    // -- Argmax (reads buf_a) --
    wire [11:0]        am_rd_addr;

    // =====================================================================
    //  WEIGHT ADDRESS MAP IN SRAM (contiguous INT8 storage)
    // =====================================================================
    localparam [23:0] WBASE_C1  = 24'd0;      // 150
    localparam [23:0] WBASE_C3  = 24'd150;    // 2400
    localparam [23:0] WBASE_C5  = 24'd2550;   // 30720
    localparam [23:0] WBASE_FC1 = 24'd33270;  // 10080
    localparam [23:0] WBASE_FC2 = 24'd43350;  // 840

    // =====================================================================
    //  FEATURE-MAP BUFFERS  (M4K dual-port RAM)
    // =====================================================================

    // ---- buf_input : 784 × 16 bit (28×28 image in Q8.8) -----------------
    wire signed [15:0] buf_in_rdata;

    dpram #(.ADDR_W(10), .DATA_W(16), .DEPTH(784)) u_buf_input (
        .clk   (clk),
        .we    (pixel_we),
        .waddr (pixel_addr),
        .wdata ({8'b0, pixel_data}),      // 8-bit pixel → Q8.8
        .raddr (c1_rd_addr[9:0]),
        .rdata (buf_in_rdata)
    );

    // ---- buf_a : 3456 × 16 bit (max after C1: 24×24×6) ------------------
    reg         buf_a_we;
    reg  [11:0] buf_a_waddr;
    reg  signed [15:0] buf_a_wdata;
    reg  [11:0] buf_a_raddr;
    wire signed [15:0] buf_a_rdata;

    dpram #(.ADDR_W(12), .DATA_W(16), .DEPTH(3456)) u_buf_a (
        .clk   (clk),
        .we    (buf_a_we),
        .waddr (buf_a_waddr),
        .wdata (buf_a_wdata),
        .raddr (buf_a_raddr),
        .rdata (buf_a_rdata)
    );

    // ---- buf_b : 864 × 16 bit (max after S2: 12×12×6) -------------------
    reg         buf_b_we;
    reg  [11:0] buf_b_waddr;
    reg  signed [15:0] buf_b_wdata;
    reg  [11:0] buf_b_raddr;
    wire signed [15:0] buf_b_rdata;

    dpram #(.ADDR_W(10), .DATA_W(16), .DEPTH(864)) u_buf_b (
        .clk   (clk),
        .we    (buf_b_we),
        .waddr (buf_b_waddr[9:0]),
        .wdata (buf_b_wdata),
        .raddr (buf_b_raddr[9:0]),
        .rdata (buf_b_rdata)
    );

    // =====================================================================
    //  BUFFER READ MUX  (only one layer active at a time)
    // =====================================================================
    always @(*) begin
        case (fsm)
            ST_RUN_S2:  buf_a_raddr = s2_rd_addr;
            ST_RUN_S4:  buf_a_raddr = s4_rd_addr;
            ST_RUN_FC1: buf_a_raddr = fc1_rd_addr;
            ST_RUN_AM:  buf_a_raddr = am_rd_addr;
            default:    buf_a_raddr = 12'd0;
        endcase
    end

    always @(*) begin
        case (fsm)
            ST_RUN_C3:  buf_b_raddr = c3_rd_addr;
            ST_RUN_C5:  buf_b_raddr = c5_rd_addr;
            ST_RUN_FC2: buf_b_raddr = fc2_rd_addr;
            default:    buf_b_raddr = 12'd0;
        endcase
    end

    // =====================================================================
    //  BUFFER WRITE MUX
    // =====================================================================
    always @(*) begin
        case (fsm)
            ST_RUN_C1:  begin buf_a_we = c1_wr_en;  buf_a_waddr = c1_wr_addr;  buf_a_wdata = c1_wr_data;  end
            ST_RUN_C3:  begin buf_a_we = c3_wr_en;  buf_a_waddr = c3_wr_addr;  buf_a_wdata = c3_wr_data;  end
            ST_RUN_C5:  begin buf_a_we = c5_wr_en;  buf_a_waddr = c5_wr_addr;  buf_a_wdata = c5_wr_data;  end
            ST_RUN_FC2: begin buf_a_we = fc2_wr_en; buf_a_waddr = fc2_wr_addr; buf_a_wdata = fc2_wr_data; end
            default:    begin buf_a_we = 1'b0;      buf_a_waddr = 12'd0;       buf_a_wdata = 16'sd0;      end
        endcase
    end

    always @(*) begin
        case (fsm)
            ST_RUN_S2:  begin buf_b_we = s2_wr_en;  buf_b_waddr = s2_wr_addr;  buf_b_wdata = s2_wr_data;  end
            ST_RUN_S4:  begin buf_b_we = s4_wr_en;  buf_b_waddr = s4_wr_addr;  buf_b_wdata = s4_wr_data;  end
            ST_RUN_FC1: begin buf_b_we = fc1_wr_en; buf_b_waddr = fc1_wr_addr; buf_b_wdata = fc1_wr_data; end
            default:    begin buf_b_we = 1'b0;      buf_b_waddr = 12'd0;       buf_b_wdata = 16'sd0;      end
        endcase
    end

    // =====================================================================
    //  SRAM WEIGHT REQUEST MUX + RESPONSE DEMUX
    // =====================================================================
    reg        sram_rd_req_r;
    reg [23:0] sram_rd_addr_r;

    assign sram_rd_req  = sram_rd_req_r;
    assign sram_rd_addr = sram_rd_addr_r;

    // Shared weight data bus. Only the active layer sees valid = 1.
    assign c1_w_data   = sram_rd_data;
    assign c3_w_data   = sram_rd_data;
    assign c5_w_data   = sram_rd_data;
    assign fc1_w_data  = sram_rd_data;
    assign fc2_w_data  = sram_rd_data;

    assign c1_w_valid  = sram_rd_valid && (fsm == ST_RUN_C1);
    assign c3_w_valid  = sram_rd_valid && (fsm == ST_RUN_C3);
    assign c5_w_valid  = sram_rd_valid && (fsm == ST_RUN_C5);
    assign fc1_w_valid = sram_rd_valid && (fsm == ST_RUN_FC1);
    assign fc2_w_valid = sram_rd_valid && (fsm == ST_RUN_FC2);

    always @(*) begin
        sram_rd_req_r  = 1'b0;
        sram_rd_addr_r = 24'd0;

        case (fsm)
            ST_RUN_C1: begin
                sram_rd_req_r  = c1_w_req;
                sram_rd_addr_r = WBASE_C1 + c1_w_addr;
            end
            ST_RUN_C3: begin
                sram_rd_req_r  = c3_w_req;
                sram_rd_addr_r = WBASE_C3 + c3_w_addr;
            end
            ST_RUN_C5: begin
                sram_rd_req_r  = c5_w_req;
                sram_rd_addr_r = WBASE_C5 + c5_w_addr;
            end
            ST_RUN_FC1: begin
                sram_rd_req_r  = fc1_w_req;
                sram_rd_addr_r = WBASE_FC1 + fc1_w_addr;
            end
            ST_RUN_FC2: begin
                sram_rd_req_r  = fc2_w_req;
                sram_rd_addr_r = WBASE_FC2 + fc2_w_addr;
            end
            default: begin
                sram_rd_req_r  = 1'b0;
                sram_rd_addr_r = 24'd0;
            end
        endcase
    end

    // =====================================================================
    //  LAYER INSTANCES
    // =====================================================================

    // ---- C1: Conv 28×28×1 → 24×24×6, kernel 5×5 -------------------------
    conv_layer #(
        .IN_SIZE   (28), .IN_CH(1), .OUT_CH(6), .KERNEL(5),
        .BIAS_FILE ("mem/conv1_bias.hex")
    ) u_c1 (
        .clk(clk), .rst_n(rst_n), .start(c1_start), .done(c1_done),
        .in_addr(c1_rd_addr), .in_data(buf_in_rdata),
        .out_addr(c1_wr_addr), .out_data(c1_wr_data), .out_we(c1_wr_en),
        .w_req(c1_w_req), .w_addr(c1_w_addr),
        .w_data(c1_w_data), .w_valid(c1_w_valid)
    );

    // ---- S2: AvgPool 24×24×6 → 12×12×6 ----------------------------------
    avgpool_layer #(
        .IN_SIZE(24), .CHANNELS(6)
    ) u_s2 (
        .clk(clk), .rst_n(rst_n), .start(s2_start), .done(s2_done),
        .in_addr(s2_rd_addr), .in_data(buf_a_rdata),
        .out_addr(s2_wr_addr), .out_data(s2_wr_data), .out_we(s2_wr_en)
    );

    // ---- C3: Conv 12×12×6 → 8×8×16, kernel 5×5 --------------------------
    conv_layer #(
        .IN_SIZE   (12), .IN_CH(6), .OUT_CH(16), .KERNEL(5),
        .BIAS_FILE ("mem/conv3_bias.hex")
    ) u_c3 (
        .clk(clk), .rst_n(rst_n), .start(c3_start), .done(c3_done),
        .in_addr(c3_rd_addr), .in_data(buf_b_rdata),
        .out_addr(c3_wr_addr), .out_data(c3_wr_data), .out_we(c3_wr_en),
        .w_req(c3_w_req), .w_addr(c3_w_addr),
        .w_data(c3_w_data), .w_valid(c3_w_valid)
    );

    // ---- S4: AvgPool 8×8×16 → 4×4×16 ------------------------------------
    avgpool_layer #(
        .IN_SIZE(8), .CHANNELS(16)
    ) u_s4 (
        .clk(clk), .rst_n(rst_n), .start(s4_start), .done(s4_done),
        .in_addr(s4_rd_addr), .in_data(buf_a_rdata),
        .out_addr(s4_wr_addr), .out_data(s4_wr_data), .out_we(s4_wr_en)
    );

    // ---- C5: Conv 4×4×16 → 1×1×120, kernel 4×4 + ReLU -------------------
    conv_layer #(
        .IN_SIZE   (4), .IN_CH(16), .OUT_CH(120), .KERNEL(4),
        .BIAS_FILE ("mem/c5_bias.hex")
    ) u_c5 (
        .clk(clk), .rst_n(rst_n), .start(c5_start), .done(c5_done),
        .in_addr(c5_rd_addr), .in_data(buf_b_rdata),
        .out_addr(c5_wr_addr), .out_data(c5_wr_data), .out_we(c5_wr_en),
        .w_req(c5_w_req), .w_addr(c5_w_addr),
        .w_data(c5_w_data), .w_valid(c5_w_valid)
    );

    // ---- FC1: 120 → 84 + ReLU -------------------------------------------
    fc_layer #(
        .IN_SIZE(120), .OUT_SIZE(84),
        .BIAS_FILE("mem/fc1_bias.hex"), .APPLY_RELU(1)
    ) u_fc1 (
        .clk(clk), .rst_n(rst_n), .start(fc1_start), .done(fc1_done),
        .in_addr(fc1_rd_addr), .in_data(buf_a_rdata),
        .out_addr(fc1_wr_addr), .out_data(fc1_wr_data), .out_we(fc1_wr_en),
        .w_req(fc1_w_req), .w_addr(fc1_w_addr),
        .w_data(fc1_w_data), .w_valid(fc1_w_valid)
    );

    // ---- FC2: 84 → 10 (no ReLU — raw logits) ----------------------------
    fc_layer #(
        .IN_SIZE(84), .OUT_SIZE(10),
        .BIAS_FILE("mem/fc2_bias.hex"), .APPLY_RELU(0)
    ) u_fc2 (
        .clk(clk), .rst_n(rst_n), .start(fc2_start), .done(fc2_done),
        .in_addr(fc2_rd_addr), .in_data(buf_b_rdata),
        .out_addr(fc2_wr_addr), .out_data(fc2_wr_data), .out_we(fc2_wr_en),
        .w_req(fc2_w_req), .w_addr(fc2_w_addr),
        .w_data(fc2_w_data), .w_valid(fc2_w_valid)
    );

    // ---- Argmax: 10 logits → 4-bit class --------------------------------
    argmax #(
        .NUM_CLASSES(10), .DATA_WIDTH(16)
    ) u_argmax (
        .clk(clk), .rst_n(rst_n), .start(am_start),
        .in_addr(am_rd_addr), .in_data(buf_a_rdata),
        .done(am_done), .class_out(digit_out)
    );

    // =====================================================================
    //  TOP FSM — sequence layers one after another
    // =====================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm       <= ST_IDLE;
            done      <= 1'b0;
            c1_start  <= 0; s2_start  <= 0; c3_start  <= 0;
            s4_start  <= 0; c5_start  <= 0;
            fc1_start <= 0; fc2_start <= 0;
            am_start  <= 0;
        end else begin
            // Default: de-assert all starts (one-shot pulses)
            c1_start  <= 0; s2_start  <= 0; c3_start  <= 0;
            s4_start  <= 0; c5_start  <= 0;
            fc1_start <= 0; fc2_start <= 0;
            am_start  <= 0;

            case (fsm)
                ST_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        c1_start <= 1'b1;
                        fsm      <= ST_RUN_C1;
                    end
                end

                ST_RUN_C1: if (c1_done) begin
                    s2_start <= 1'b1;  fsm <= ST_RUN_S2;
                end

                ST_RUN_S2: if (s2_done) begin
                    c3_start <= 1'b1;  fsm <= ST_RUN_C3;
                end

                ST_RUN_C3: if (c3_done) begin
                    s4_start <= 1'b1;  fsm <= ST_RUN_S4;
                end

                ST_RUN_S4: if (s4_done) begin
                    c5_start <= 1'b1;  fsm <= ST_RUN_C5;
                end

                ST_RUN_C5: if (c5_done) begin
                    fc1_start <= 1'b1; fsm <= ST_RUN_FC1;
                end

                ST_RUN_FC1: if (fc1_done) begin
                    fc2_start <= 1'b1; fsm <= ST_RUN_FC2;
                end

                ST_RUN_FC2: if (fc2_done) begin
                    am_start <= 1'b1;  fsm <= ST_RUN_AM;
                end

                ST_RUN_AM: if (am_done) begin
                    done <= 1'b1;  fsm <= ST_DONE;
                end

                ST_DONE: begin
                    if (start) begin
                        done     <= 1'b0;
                        c1_start <= 1'b1;
                        fsm      <= ST_RUN_C1;
                    end
                end
            endcase
        end
    end

endmodule
