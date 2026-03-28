// ============================================================================
// LeNet-5 Top-Level Module
// Inference-only CNN for MNIST 28×28 digit recognition.
//
// Architecture matches Python model.py:
//   Conv2d(1→6, k=5) + ReLU → AvgPool(2) →
//   Conv2d(6→16, k=5) + ReLU → AvgPool(2) →
//   Conv2d(16→120, k=4) + ReLU → flatten →
//   Linear(120→84) + ReLU → Linear(84→10) → Argmax
//
// Interface:
//   - Write 28×28 = 784 pixels (8-bit) to the input buffer via pixel bus
//   - Assert 'start' for one clock cycle
//   - Wait for 'done'; digit_out holds the recognized class (0-9)
//
// Internal data flow:
//   buf_input → C1(conv5×5,1→6)     → buf_a  (24×24×6 = 3456)
//   buf_a     → S2(avgpool2×2)       → buf_b  (12×12×6 = 864)
//   buf_b     → C3(conv5×5,6→16)    → buf_a  (8×8×16  = 1024)
//   buf_a     → S4(avgpool2×2)       → buf_b  (4×4×16  = 256)
//   buf_b     → C5(conv4×4,16→120)  → buf_a  (1×1×120 = 120)
//   buf_a     → FC1(120→84)+ReLU    → buf_b  (84)
//   buf_b     → FC2(84→10)          → buf_a  (10)
//   buf_a     → Argmax              → digit_out
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
    output wire [3:0]  digit_out
);

    // =====================================================================
    //  INTERNAL BUFFERS
    // =====================================================================
    // buf_input : 28×28 = 784 words (stores image in Q8.8)
    // buf_a     : max 24×24×6 = 3456 words (after C1)
    // buf_b     : max 12×12×6 = 864  words (after S2)
    reg signed [15:0] buf_input [0:783];
    reg signed [15:0] buf_a     [0:3455];
    reg signed [15:0] buf_b     [0:863];

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
    wire signed [15:0] c1_rd_data;
    wire [11:0]        c1_wr_addr;
    wire signed [15:0] c1_wr_data;
    wire               c1_wr_en;

    // -- S2 (reads buf_a, writes buf_b) --
    wire [11:0]        s2_rd_addr;
    wire signed [15:0] s2_rd_data;
    wire [11:0]        s2_wr_addr;
    wire signed [15:0] s2_wr_data;
    wire               s2_wr_en;

    // -- C3 (reads buf_b, writes buf_a) --
    wire [11:0]        c3_rd_addr;
    wire signed [15:0] c3_rd_data;
    wire [11:0]        c3_wr_addr;
    wire signed [15:0] c3_wr_data;
    wire               c3_wr_en;

    // -- S4 (reads buf_a, writes buf_b) --
    wire [11:0]        s4_rd_addr;
    wire signed [15:0] s4_rd_data;
    wire [11:0]        s4_wr_addr;
    wire signed [15:0] s4_wr_data;
    wire               s4_wr_en;

    // -- C5 (reads buf_b, writes buf_a) --
    wire [11:0]        c5_rd_addr;
    wire signed [15:0] c5_rd_data;
    wire [11:0]        c5_wr_addr;
    wire signed [15:0] c5_wr_data;
    wire               c5_wr_en;

    // -- FC1 (reads buf_a, writes buf_b) --
    wire [11:0]        fc1_rd_addr;
    wire signed [15:0] fc1_rd_data;
    wire [11:0]        fc1_wr_addr;
    wire signed [15:0] fc1_wr_data;
    wire               fc1_wr_en;

    // -- FC2 (reads buf_b, writes buf_a) --
    wire [11:0]        fc2_rd_addr;
    wire signed [15:0] fc2_rd_data;
    wire [11:0]        fc2_wr_addr;
    wire signed [15:0] fc2_wr_data;
    wire               fc2_wr_en;

    // -- Argmax (reads buf_a) --
    wire [11:0]        am_rd_addr;
    wire signed [15:0] am_rd_data;

    // =====================================================================
    //  BUFFER READ (combinational)
    // =====================================================================
    assign c1_rd_data  = buf_input[c1_rd_addr];
    assign s2_rd_data  = buf_a[s2_rd_addr];
    assign c3_rd_data  = buf_b[c3_rd_addr];
    assign s4_rd_data  = buf_a[s4_rd_addr];
    assign c5_rd_data  = buf_b[c5_rd_addr];
    assign fc1_rd_data = buf_a[fc1_rd_addr];
    assign fc2_rd_data = buf_b[fc2_rd_addr];
    assign am_rd_data  = buf_a[am_rd_addr];

    // =====================================================================
    //  BUFFER WRITE (clocked)
    // =====================================================================
    // --- Pixel load into buf_input (convert 8-bit → Q8.8) ----------------
    always @(posedge clk) begin
        if (pixel_we)
            buf_input[pixel_addr] <= {8'b0, pixel_data};  // 0.xxxx in Q8.8
    end

    // --- buf_a writes (from C1, C3, C5, FC2) -----------------------------
    always @(posedge clk) begin
        if (c1_wr_en)  buf_a[c1_wr_addr]  <= c1_wr_data;
        if (c3_wr_en)  buf_a[c3_wr_addr]  <= c3_wr_data;
        if (c5_wr_en)  buf_a[c5_wr_addr]  <= c5_wr_data;
        if (fc2_wr_en) buf_a[fc2_wr_addr] <= fc2_wr_data;
    end

    // --- buf_b writes (from S2, S4, FC1) ---------------------------------
    always @(posedge clk) begin
        if (s2_wr_en)  buf_b[s2_wr_addr]  <= s2_wr_data;
        if (s4_wr_en)  buf_b[s4_wr_addr]  <= s4_wr_data;
        if (fc1_wr_en) buf_b[fc1_wr_addr] <= fc1_wr_data;
    end

    // =====================================================================
    //  LAYER INSTANCES
    // =====================================================================

    // ---- C1: Conv 28×28×1 → 24×24×6, kernel 5×5 -------------------------
    conv_layer #(
        .IN_SIZE     (28),
        .IN_CH       (1),
        .OUT_CH      (6),
        .KERNEL      (5),
        .WEIGHT_FILE ("mem/conv1_weights.hex"),
        .BIAS_FILE   ("mem/conv1_bias.hex")
    ) u_c1 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (c1_start),
        .done     (c1_done),
        .in_addr  (c1_rd_addr),
        .in_data  (c1_rd_data),
        .out_addr (c1_wr_addr),
        .out_data (c1_wr_data),
        .out_we   (c1_wr_en)
    );

    // ---- S2: AvgPool 24×24×6 → 12×12×6 ----------------------------------
    avgpool_layer #(
        .IN_SIZE  (24),
        .CHANNELS (6)
    ) u_s2 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (s2_start),
        .done     (s2_done),
        .in_addr  (s2_rd_addr),
        .in_data  (s2_rd_data),
        .out_addr (s2_wr_addr),
        .out_data (s2_wr_data),
        .out_we   (s2_wr_en)
    );

    // ---- C3: Conv 12×12×6 → 8×8×16, kernel 5×5 --------------------------
    conv_layer #(
        .IN_SIZE     (12),
        .IN_CH       (6),
        .OUT_CH      (16),
        .KERNEL      (5),
        .WEIGHT_FILE ("mem/conv3_weights.hex"),
        .BIAS_FILE   ("mem/conv3_bias.hex")
    ) u_c3 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (c3_start),
        .done     (c3_done),
        .in_addr  (c3_rd_addr),
        .in_data  (c3_rd_data),
        .out_addr (c3_wr_addr),
        .out_data (c3_wr_data),
        .out_we   (c3_wr_en)
    );

    // ---- S4: AvgPool 8×8×16 → 4×4×16 ------------------------------------
    avgpool_layer #(
        .IN_SIZE  (8),
        .CHANNELS (16)
    ) u_s4 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (s4_start),
        .done     (s4_done),
        .in_addr  (s4_rd_addr),
        .in_data  (s4_rd_data),
        .out_addr (s4_wr_addr),
        .out_data (s4_wr_data),
        .out_we   (s4_wr_en)
    );

    // ---- C5: Conv 4×4×16 → 1×1×120, kernel 4×4 + ReLU -------------------
    conv_layer #(
        .IN_SIZE     (4),
        .IN_CH       (16),
        .OUT_CH      (120),
        .KERNEL      (4),
        .WEIGHT_FILE ("mem/c5_weights.hex"),
        .BIAS_FILE   ("mem/c5_bias.hex")
    ) u_c5 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (c5_start),
        .done     (c5_done),
        .in_addr  (c5_rd_addr),
        .in_data  (c5_rd_data),
        .out_addr (c5_wr_addr),
        .out_data (c5_wr_data),
        .out_we   (c5_wr_en)
    );

    // ---- FC1: 120 → 84 + ReLU -------------------------------------------
    fc_layer #(
        .IN_SIZE     (120),
        .OUT_SIZE    (84),
        .WEIGHT_FILE ("mem/fc1_weights.hex"),
        .BIAS_FILE   ("mem/fc1_bias.hex"),
        .APPLY_RELU  (1)
    ) u_fc1 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (fc1_start),
        .done     (fc1_done),
        .in_addr  (fc1_rd_addr),
        .in_data  (fc1_rd_data),
        .out_addr (fc1_wr_addr),
        .out_data (fc1_wr_data),
        .out_we   (fc1_wr_en)
    );

    // ---- FC2: 84 → 10 (no ReLU — raw logits) ----------------------------
    fc_layer #(
        .IN_SIZE     (84),
        .OUT_SIZE    (10),
        .WEIGHT_FILE ("mem/fc2_weights.hex"),
        .BIAS_FILE   ("mem/fc2_bias.hex"),
        .APPLY_RELU  (0)
    ) u_fc2 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (fc2_start),
        .done     (fc2_done),
        .in_addr  (fc2_rd_addr),
        .in_data  (fc2_rd_data),
        .out_addr (fc2_wr_addr),
        .out_data (fc2_wr_data),
        .out_we   (fc2_wr_en)
    );

    // ---- Argmax: 10 logits → 4-bit class --------------------------------
    argmax #(
        .NUM_CLASSES (10),
        .DATA_WIDTH  (16)
    ) u_argmax (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (am_start),
        .in_addr   (am_rd_addr),
        .in_data   (am_rd_data),
        .done      (am_done),
        .class_out (digit_out)
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

                ST_RUN_C1: begin
                    if (c1_done) begin
                        s2_start <= 1'b1;
                        fsm      <= ST_RUN_S2;
                    end
                end

                ST_RUN_S2: begin
                    if (s2_done) begin
                        c3_start <= 1'b1;
                        fsm      <= ST_RUN_C3;
                    end
                end

                ST_RUN_C3: begin
                    if (c3_done) begin
                        s4_start <= 1'b1;
                        fsm      <= ST_RUN_S4;
                    end
                end

                ST_RUN_S4: begin
                    if (s4_done) begin
                        c5_start <= 1'b1;
                        fsm      <= ST_RUN_C5;
                    end
                end

                ST_RUN_C5: begin
                    if (c5_done) begin
                        fc1_start <= 1'b1;
                        fsm       <= ST_RUN_FC1;
                    end
                end

                ST_RUN_FC1: begin
                    if (fc1_done) begin
                        fc2_start <= 1'b1;
                        fsm       <= ST_RUN_FC2;
                    end
                end

                ST_RUN_FC2: begin
                    if (fc2_done) begin
                        am_start <= 1'b1;
                        fsm      <= ST_RUN_AM;
                    end
                end

                ST_RUN_AM: begin
                    if (am_done) begin
                        done <= 1'b1;
                        fsm  <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    // Hold done & digit_out until next start
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
