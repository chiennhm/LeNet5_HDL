// ============================================================================
// LeNet-5 Top-Level Module
// Inference-only CNN for MNIST 14×14 digit recognition.
//
// Interface:
//   - Write 14×14 = 196 pixels (8-bit) to the input buffer via pixel bus
//   - Assert 'start' for one clock cycle
//   - Wait for 'done'; digit_out holds the recognized class (0-9)
//
// Internal data flow:
//   buf_input → C1(conv5×5,6) → buf_a
//   buf_a     → S2(pool2×2)   → buf_b
//   buf_b     → C3(conv5×5,16)→ buf_a
//   buf_a     → FC1(16→120)   → buf_b
//   buf_b     → FC2(120→84)   → buf_a
//   buf_a     → FC3(84→10)    → buf_b
//   buf_b     → Argmax        → digit_out
// ============================================================================
module lenet5_top (
    input  wire        clk,
    input  wire        rst_n,

    // ---- Image load interface -------------------------------------------
    input  wire [7:0]  pixel_data,    // 8-bit grayscale pixel
    input  wire [7:0]  pixel_addr,    // address 0..195
    input  wire        pixel_we,      // write-enable

    // ---- Control & result -----------------------------------------------
    input  wire        start,
    output reg         done,
    output wire [3:0]  digit_out
);

    // =====================================================================
    //  INTERNAL BUFFERS
    // =====================================================================
    // buf_input : 14×14 = 196 words (stores image in Q8.8)
    // buf_a     : max 10×10×6 = 600 words
    // buf_b     : max 5×5×6   = 150 words
    reg signed [15:0] buf_input [0:195];
    reg signed [15:0] buf_a     [0:599];
    reg signed [15:0] buf_b     [0:149];

    // =====================================================================
    //  FSM
    // =====================================================================
    localparam ST_IDLE      = 4'd0,
               ST_RUN_C1    = 4'd1,
               ST_RUN_S2    = 4'd2,
               ST_RUN_C3    = 4'd3,
               ST_RUN_FC1   = 4'd4,
               ST_RUN_FC2   = 4'd5,
               ST_RUN_FC3   = 4'd6,
               ST_RUN_AM    = 4'd7,
               ST_DONE      = 4'd8;
    reg [3:0] fsm;

    // =====================================================================
    //  LAYER START / DONE SIGNALS
    // =====================================================================
    reg  c1_start, s2_start, c3_start;
    reg  fc1_start, fc2_start, fc3_start, am_start;
    wire c1_done,  s2_done,  c3_done;
    wire fc1_done, fc2_done, fc3_done, am_done;

    // =====================================================================
    //  LAYER ↔ BUFFER ADDRESS / DATA WIRES
    // =====================================================================
    // -- C1 (reads buf_input, writes buf_a) --
    wire [9:0]         c1_rd_addr;
    wire signed [15:0] c1_rd_data;
    wire [9:0]         c1_wr_addr;
    wire signed [15:0] c1_wr_data;
    wire               c1_wr_en;

    // -- S2 (reads buf_a, writes buf_b) --
    wire [9:0]         s2_rd_addr;
    wire signed [15:0] s2_rd_data;
    wire [9:0]         s2_wr_addr;
    wire signed [15:0] s2_wr_data;
    wire               s2_wr_en;

    // -- C3 (reads buf_b, writes buf_a) --
    wire [9:0]         c3_rd_addr;
    wire signed [15:0] c3_rd_data;
    wire [9:0]         c3_wr_addr;
    wire signed [15:0] c3_wr_data;
    wire               c3_wr_en;

    // -- FC1 (reads buf_a, writes buf_b) --
    wire [9:0]         fc1_rd_addr;
    wire signed [15:0] fc1_rd_data;
    wire [9:0]         fc1_wr_addr;
    wire signed [15:0] fc1_wr_data;
    wire               fc1_wr_en;

    // -- FC2 (reads buf_b, writes buf_a) --
    wire [9:0]         fc2_rd_addr;
    wire signed [15:0] fc2_rd_data;
    wire [9:0]         fc2_wr_addr;
    wire signed [15:0] fc2_wr_data;
    wire               fc2_wr_en;

    // -- FC3 (reads buf_a, writes buf_b) --
    wire [9:0]         fc3_rd_addr;
    wire signed [15:0] fc3_rd_data;
    wire [9:0]         fc3_wr_addr;
    wire signed [15:0] fc3_wr_data;
    wire               fc3_wr_en;

    // -- Argmax (reads buf_b) --
    wire [9:0]         am_rd_addr;
    wire signed [15:0] am_rd_data;

    // =====================================================================
    //  BUFFER READ (combinational)
    // =====================================================================
    assign c1_rd_data  = buf_input[c1_rd_addr];
    assign s2_rd_data  = buf_a[s2_rd_addr];
    assign c3_rd_data  = buf_b[c3_rd_addr];
    assign fc1_rd_data = buf_a[fc1_rd_addr];
    assign fc2_rd_data = buf_b[fc2_rd_addr];
    assign fc3_rd_data = buf_a[fc3_rd_addr];
    assign am_rd_data  = buf_b[am_rd_addr];

    // =====================================================================
    //  BUFFER WRITE (clocked)
    // =====================================================================
    // --- Pixel load into buf_input (convert 8-bit → Q8.8) ----------------
    always @(posedge clk) begin
        if (pixel_we)
            buf_input[pixel_addr] <= {8'b0, pixel_data};  // 0.xxxx in Q8.8
    end

    // --- buf_a writes (from C1, C3, FC2) ---------------------------------
    always @(posedge clk) begin
        if (c1_wr_en)  buf_a[c1_wr_addr]  <= c1_wr_data;
        if (c3_wr_en)  buf_a[c3_wr_addr]  <= c3_wr_data;
        if (fc2_wr_en) buf_a[fc2_wr_addr] <= fc2_wr_data;
    end

    // --- buf_b writes (from S2, FC1, FC3) --------------------------------
    always @(posedge clk) begin
        if (s2_wr_en)  buf_b[s2_wr_addr]  <= s2_wr_data;
        if (fc1_wr_en) buf_b[fc1_wr_addr] <= fc1_wr_data;
        if (fc3_wr_en) buf_b[fc3_wr_addr] <= fc3_wr_data;
    end

    // =====================================================================
    //  LAYER INSTANCES
    // =====================================================================

    // ---- C1: Conv 14×14×1 → 10×10×6, kernel 5×5 -------------------------
    conv_layer #(
        .IN_SIZE     (14),
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

    // ---- S2: MaxPool 10×10×6 → 5×5×6 ------------------------------------
    maxpool_layer #(
        .IN_SIZE  (10),
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

    // ---- C3: Conv 5×5×6 → 1×1×16, kernel 5×5 ----------------------------
    conv_layer #(
        .IN_SIZE     (5),
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

    // ---- FC1: 16 → 120 + ReLU -------------------------------------------
    fc_layer #(
        .IN_SIZE     (16),
        .OUT_SIZE    (120),
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

    // ---- FC2: 120 → 84 + ReLU -------------------------------------------
    fc_layer #(
        .IN_SIZE     (120),
        .OUT_SIZE    (84),
        .WEIGHT_FILE ("mem/fc2_weights.hex"),
        .BIAS_FILE   ("mem/fc2_bias.hex"),
        .APPLY_RELU  (1)
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

    // ---- FC3: 84 → 10 (no ReLU — raw logits) ----------------------------
    fc_layer #(
        .IN_SIZE     (84),
        .OUT_SIZE    (10),
        .WEIGHT_FILE ("mem/fc3_weights.hex"),
        .BIAS_FILE   ("mem/fc3_bias.hex"),
        .APPLY_RELU  (0)
    ) u_fc3 (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (fc3_start),
        .done     (fc3_done),
        .in_addr  (fc3_rd_addr),
        .in_data  (fc3_rd_data),
        .out_addr (fc3_wr_addr),
        .out_data (fc3_wr_data),
        .out_we   (fc3_wr_en)
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
            fc1_start <= 0; fc2_start <= 0; fc3_start <= 0;
            am_start  <= 0;
        end else begin
            // Default: de-assert all starts (one-shot pulses)
            c1_start  <= 0; s2_start  <= 0; c3_start  <= 0;
            fc1_start <= 0; fc2_start <= 0; fc3_start <= 0;
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
                        fc3_start <= 1'b1;
                        fsm       <= ST_RUN_FC3;
                    end
                end

                ST_RUN_FC3: begin
                    if (fc3_done) begin
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
