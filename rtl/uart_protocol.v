// ============================================================================
// UART Binary Protocol Handler for LeNet-5 DE2
//
// Implements the frame protocol defined in uart_loader.py:
//
//   Host→Device : [SOF=0xA5][CMD][SEQ][LEN_L][LEN_H][PAYLOAD…][CRC_L][CRC_H]
//   Device→Host : [SOF=0x5A][SEQ][STATUS][LEN_L][LEN_H][PAYLOAD…][CRC_L][CRC_H]
//
// CRC: CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)
//
// Commands:
//   0x00  PING            → empty response
//   0x01  WRITE_SRAM      → writes payload data to external SRAM
//   0x02  START_INFERENCE  → asserts start pulse
//   0x03  GET_STATUS       → returns [done, digit]
//   0x04  WRITE_IMAGE     → writes pixel data to image input buffer
//
// Max buffered payload: 512 bytes (supports chunk_size up to 507).
// ============================================================================
module uart_protocol #(
    parameter MAX_PAYLOAD = 512     // payload buffer depth
)(
    input  wire       clk,
    input  wire       rst_n,

    // --- UART byte-level interface ----------------------------------------
    input  wire [7:0] rx_data,
    input  wire       rx_valid,
    output reg  [7:0] tx_data,
    output reg        tx_start,
    input  wire       tx_busy,

    // --- SRAM write interface ---------------------------------------------
    output reg        sram_wr_req,
    output reg [23:0] sram_wr_addr,
    output reg  [7:0] sram_wr_data,
    input  wire       sram_wr_done,

    // --- LeNet core control -----------------------------------------------
    output reg        core_start,       // one-cycle pulse
    input  wire       core_done,
    input  wire [3:0] core_digit,

    // --- Image buffer write interface -------------------------------------
    output reg        img_wr_en,
    output reg  [9:0] img_wr_addr,
    output reg  [7:0] img_wr_data
);

    // =====================================================================
    //  Constants
    // =====================================================================
    localparam HOST_SOF = 8'hA5;
    localparam DEV_SOF  = 8'h5A;

    localparam CMD_PING            = 8'h00;
    localparam CMD_WRITE_SRAM      = 8'h01;
    localparam CMD_START_INFERENCE = 8'h02;
    localparam CMD_GET_STATUS      = 8'h03;
    localparam CMD_WRITE_IMAGE     = 8'h04;

    // =====================================================================
    //  FSM states
    // =====================================================================
    localparam [4:0]
        RX_IDLE     = 5'd0,
        RX_CMD      = 5'd1,
        RX_SEQ      = 5'd2,
        RX_LEN_L    = 5'd3,
        RX_LEN_H    = 5'd4,
        RX_PAYLOAD  = 5'd5,
        RX_CRC_L    = 5'd6,
        RX_CRC_H    = 5'd7,
        DISPATCH    = 5'd8,
        WR_INIT     = 5'd9,
        WR_READ_BUF = 5'd10,
        WR_ISSUE    = 5'd11,
        WR_WAIT     = 5'd12,
        TX_SOF      = 5'd13,
        TX_SEQ      = 5'd14,
        TX_STATUS   = 5'd15,
        TX_LEN_L    = 5'd16,
        TX_LEN_H    = 5'd17,
        TX_PAYLOAD  = 5'd18,
        TX_CRC_L    = 5'd19,
        TX_CRC_H    = 5'd20,
        TX_DONE     = 5'd21,
        IMG_INIT     = 5'd22,
        IMG_READ_HDR = 5'd23,
        IMG_WRITE    = 5'd24;

    reg [4:0] state;

    // =====================================================================
    //  RX frame registers
    // =====================================================================
    reg [7:0]  rx_cmd;
    reg [7:0]  rx_seq;
    reg [15:0] rx_len;          // payload length
    reg [15:0] rx_cnt;          // payload byte counter
    reg [15:0] rx_crc_rx;       // CRC received from host
    reg [15:0] rx_crc_calc;     // CRC computed locally

    // =====================================================================
    //  Payload buffer (dual-port RAM style, using register array)
    //  At 9600 baud the data rate is only ~960 B/s, so 512 regs are fine
    //  and avoid an extra dpram instance.
    // =====================================================================
    reg [7:0] payload_buf [0:MAX_PAYLOAD-1];
    reg [8:0] buf_waddr;
    reg [8:0] buf_raddr;
    reg [7:0] buf_rdata;

    // Read from payload buffer (synchronous)
    always @(posedge clk) begin
        buf_rdata <= payload_buf[buf_raddr];
    end

    // =====================================================================
    //  TX response registers
    // =====================================================================
    reg [7:0]  resp_status;
    reg [15:0] resp_len;
    reg [7:0]  resp_buf [0:3];  // max 2 bytes response payload
    reg [15:0] resp_cnt;
    reg [15:0] tx_crc;

    // =====================================================================
    //  WRITE_SRAM helpers
    // =====================================================================
    reg [23:0] wr_base_addr;
    reg [15:0] wr_size;
    reg [15:0] wr_idx;

    // =====================================================================
    //  WRITE_IMAGE helpers
    // =====================================================================
    reg [15:0] img_offset;
    reg [15:0] img_size;
    reg [15:0] img_idx;

    // =====================================================================
    //  CRC-16/CCITT-FALSE — one-byte update (combinational)
    // =====================================================================
    function [15:0] crc16_byte;
        input [15:0] crc_in;
        input [7:0]  data;
        reg   [15:0] c;
        integer i;
        begin
            c = crc_in ^ ({8'd0, data} << 8);
            for (i = 0; i < 8; i = i + 1) begin
                if (c[15])
                    c = (c << 1) ^ 16'h1021;
                else
                    c = c << 1;
            end
            crc16_byte = c;
        end
    endfunction

    // =====================================================================
    //  Main FSM
    // =====================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= RX_IDLE;
            rx_cmd       <= 8'd0;
            rx_seq       <= 8'd0;
            rx_len       <= 16'd0;
            rx_cnt       <= 16'd0;
            rx_crc_rx    <= 16'd0;
            rx_crc_calc  <= 16'hFFFF;
            buf_waddr    <= 9'd0;
            buf_raddr    <= 9'd0;
            resp_status  <= 8'd0;
            resp_len     <= 16'd0;
            resp_cnt     <= 16'd0;
            tx_crc       <= 16'hFFFF;
            tx_data      <= 8'd0;
            tx_start     <= 1'b0;
            sram_wr_req  <= 1'b0;
            sram_wr_addr <= 24'd0;
            sram_wr_data <= 8'd0;
            core_start   <= 1'b0;
            wr_base_addr <= 24'd0;
            wr_size      <= 16'd0;
            wr_idx       <= 16'd0;
            img_wr_en    <= 1'b0;
            img_wr_addr  <= 10'd0;
            img_wr_data  <= 8'd0;
            img_offset   <= 16'd0;
            img_size     <= 16'd0;
            img_idx      <= 16'd0;
        end else begin
            // defaults
            tx_start    <= 1'b0;
            core_start  <= 1'b0;
            sram_wr_req <= 1'b0;
            img_wr_en   <= 1'b0;

            case (state)
                // =============================================================
                //  RX path — receive a host frame
                // =============================================================
                RX_IDLE: begin
                    if (rx_valid && rx_data == HOST_SOF) begin
                        rx_crc_calc <= 16'hFFFF;
                        state       <= RX_CMD;
                    end
                end

                RX_CMD: begin
                    if (rx_valid) begin
                        rx_cmd      <= rx_data;
                        rx_crc_calc <= crc16_byte(rx_crc_calc, rx_data);
                        state       <= RX_SEQ;
                    end
                end

                RX_SEQ: begin
                    if (rx_valid) begin
                        rx_seq      <= rx_data;
                        rx_crc_calc <= crc16_byte(rx_crc_calc, rx_data);
                        state       <= RX_LEN_L;
                    end
                end

                RX_LEN_L: begin
                    if (rx_valid) begin
                        rx_len[7:0] <= rx_data;
                        rx_crc_calc <= crc16_byte(rx_crc_calc, rx_data);
                        state       <= RX_LEN_H;
                    end
                end

                RX_LEN_H: begin
                    if (rx_valid) begin
                        rx_len[15:8] <= rx_data;
                        rx_crc_calc  <= crc16_byte(rx_crc_calc, rx_data);
                        rx_cnt       <= 16'd0;
                        buf_waddr    <= 9'd0;
                        if ({rx_data, rx_len[7:0]} == 16'd0)
                            state <= RX_CRC_L;    // no payload
                        else
                            state <= RX_PAYLOAD;
                    end
                end

                RX_PAYLOAD: begin
                    if (rx_valid) begin
                        rx_crc_calc <= crc16_byte(rx_crc_calc, rx_data);
                        if (buf_waddr < MAX_PAYLOAD)
                            payload_buf[buf_waddr] <= rx_data;
                        buf_waddr <= buf_waddr + 9'd1;
                        rx_cnt    <= rx_cnt + 16'd1;
                        if (rx_cnt + 16'd1 == rx_len)
                            state <= RX_CRC_L;
                    end
                end

                RX_CRC_L: begin
                    if (rx_valid) begin
                        rx_crc_rx[7:0] <= rx_data;
                        state          <= RX_CRC_H;
                    end
                end

                RX_CRC_H: begin
                    if (rx_valid) begin
                        rx_crc_rx[15:8] <= rx_data;
                        state           <= DISPATCH;
                    end
                end

                // =============================================================
                //  DISPATCH — verify CRC, execute command
                // =============================================================
                DISPATCH: begin
                    if (rx_crc_calc != {rx_crc_rx[15:8], rx_crc_rx[7:0]}) begin
                        // CRC mismatch — respond with error status
                        resp_status <= 8'hFF;
                        resp_len    <= 16'd0;
                        state       <= TX_SOF;
                    end else begin
                        resp_status <= 8'h00;

                        case (rx_cmd)
                            CMD_PING: begin
                                resp_len <= 16'd0;
                                state    <= TX_SOF;
                            end

                            CMD_WRITE_SRAM: begin
                                // payload[0..2] = addr,  [3..4] = size
                                buf_raddr <= 9'd0;
                                state     <= WR_INIT;
                            end

                            CMD_START_INFERENCE: begin
                                core_start <= 1'b1;
                                resp_len   <= 16'd0;
                                state      <= TX_SOF;
                            end

                            CMD_GET_STATUS: begin
                                resp_len    <= 16'd2;
                                resp_buf[0] <= {7'd0, core_done};
                                resp_buf[1] <= {4'd0, core_digit};
                                state       <= TX_SOF;
                            end

                            CMD_WRITE_IMAGE: begin
                                // payload[0..1] = offset, [2..3] = size
                                buf_raddr <= 9'd0;
                                state     <= IMG_INIT;
                            end

                            default: begin
                                resp_status <= 8'h01;   // unknown command
                                resp_len    <= 16'd0;
                                state       <= TX_SOF;
                            end
                        endcase
                    end
                end

                // =============================================================
                //  WRITE_SRAM — stream buffered payload into SRAM
                // =============================================================
                WR_INIT: begin
                    // Wait one cycle for buf_rdata of addr byte 0
                    buf_raddr <= 9'd1;
                    state     <= WR_INIT + 5'd1;  // trick: falls through to a unique handler
                    // Actually, let me use explicit sub-states for clarity.
                    // We'll read the 5 header bytes from the buffer over
                    // several cycles, then iterate over data bytes.

                    // Cycle 0: buf_raddr=0 was set in DISPATCH, read addr[0]
                    // We need to clock through to get buf_rdata.
                    // Let's read all 5 bytes in a mini-sequence.
                    wr_idx <= 16'd0;
                    state  <= WR_READ_BUF;
                end

                // We re-use WR_READ_BUF to read the 5-byte header from the
                // payload buffer (addr[2:0], size[1:0]).  wr_idx counts 0..4.
                WR_READ_BUF: begin
                    // buf_rdata is valid for (buf_raddr - 1) due to sync read.
                    case (wr_idx)
                        16'd0: begin
                            // buf_raddr was set to 0 two cycles ago → rdata valid
                            wr_base_addr[7:0] <= buf_rdata;
                            buf_raddr <= 9'd2;
                        end
                        16'd1: begin
                            wr_base_addr[15:8] <= buf_rdata;
                            buf_raddr <= 9'd3;
                        end
                        16'd2: begin
                            wr_base_addr[23:16] <= buf_rdata;
                            buf_raddr <= 9'd4;
                        end
                        16'd3: begin
                            wr_size[7:0] <= buf_rdata;
                            buf_raddr <= 9'd5;    // Request first data byte NOW
                        end
                        16'd4: begin
                            wr_size[15:8] <= buf_rdata;
                            wr_idx <= 16'd0;      // reuse as data index
                            buf_raddr <= 9'd6;    // Request second data byte NOW
                            // If size is 0, skip directly to response
                            if ({buf_rdata, wr_size[7:0]} == 16'd0) begin
                                resp_len    <= 16'd2;
                                resp_buf[0] <= wr_size[7:0];
                                resp_buf[1] <= buf_rdata;
                                state       <= TX_SOF;
                            end else begin
                                state <= WR_ISSUE;
                            end
                        end
                        default: ;
                    endcase
                    if (wr_idx < 16'd4)
                        wr_idx <= wr_idx + 16'd1;
                end

                WR_ISSUE: begin
                    // buf_rdata holds the current data byte
                    sram_wr_req  <= 1'b1;
                    sram_wr_addr <= wr_base_addr + {8'd0, wr_idx};
                    sram_wr_data <= buf_rdata;
                    buf_raddr    <= 9'd5 + wr_idx[8:0] + 9'd1; // Prefetch next byte
                    state        <= WR_WAIT;
                end

                WR_WAIT: begin
                    if (sram_wr_done) begin
                        wr_idx <= wr_idx + 16'd1;
                        if (wr_idx + 16'd1 == wr_size) begin
                            // All bytes written — send response
                            resp_len    <= 16'd2;
                            resp_buf[0] <= wr_size[7:0];
                            resp_buf[1] <= wr_size[15:8];
                            state       <= TX_SOF;
                        end else begin
                            state     <= WR_ISSUE;
                        end
                    end
                end

                // =============================================================
                //  WRITE_IMAGE — stream buffered payload into image buffer
                //  Payload: [OFFSET_L][OFFSET_H][SIZE_L][SIZE_H][DATA...]
                //  dpram write is single-cycle, no handshake needed.
                // =============================================================
                IMG_INIT: begin
                    buf_raddr <= 9'd1;
                    img_idx   <= 16'd0;
                    state     <= IMG_READ_HDR;
                end

                IMG_READ_HDR: begin
                    case (img_idx)
                        16'd0: begin
                            img_offset[7:0] <= buf_rdata;
                            buf_raddr <= 9'd2;
                        end
                        16'd1: begin
                            img_offset[15:8] <= buf_rdata;
                            buf_raddr <= 9'd3;
                        end
                        16'd2: begin
                            img_size[7:0] <= buf_rdata;
                            buf_raddr <= 9'd5;  // Request first data byte NOW
                        end
                        16'd3: begin
                            img_size[15:8] <= buf_rdata;
                            img_idx <= 16'd0;
                            buf_raddr <= 9'd6;  // Request second data byte NOW
                            if ({buf_rdata, img_size[7:0]} == 16'd0) begin
                                resp_len    <= 16'd2;
                                resp_buf[0] <= img_size[7:0];
                                resp_buf[1] <= buf_rdata;
                                state       <= TX_SOF;
                            end else begin
                                state <= IMG_WRITE;
                            end
                        end
                        default: ;
                    endcase
                    if (img_idx < 16'd3)
                        img_idx <= img_idx + 16'd1;
                end

                IMG_WRITE: begin
                    // Write one pixel per cycle
                    img_wr_en   <= 1'b1;
                    img_wr_addr <= img_offset[9:0] + img_idx[9:0];
                    img_wr_data <= buf_rdata;
                    img_idx     <= img_idx + 16'd1;
                    if (img_idx + 16'd1 == img_size) begin
                        resp_len    <= 16'd2;
                        resp_buf[0] <= img_size[7:0];
                        resp_buf[1] <= img_size[15:8];
                        state       <= TX_SOF;
                    end else begin
                        buf_raddr <= 9'd5 + img_idx[8:0] + 9'd2; // Request next+1 byte
                        state     <= IMG_WRITE;
                    end
                end

                // =============================================================
                //  TX path — build and send response frame
                // =============================================================
                TX_SOF: begin
                    if (!tx_busy) begin
                        tx_data  <= DEV_SOF;
                        tx_start <= 1'b1;
                        tx_crc   <= 16'hFFFF;
                        resp_cnt <= 16'd0;
                        state    <= TX_SEQ;
                    end
                end

                TX_SEQ: begin
                    if (!tx_busy && !tx_start) begin
                        tx_data  <= rx_seq;
                        tx_start <= 1'b1;
                        tx_crc   <= crc16_byte(tx_crc, rx_seq);
                        state    <= TX_STATUS;
                    end
                end

                TX_STATUS: begin
                    if (!tx_busy && !tx_start) begin
                        tx_data  <= resp_status;
                        tx_start <= 1'b1;
                        tx_crc   <= crc16_byte(tx_crc, resp_status);
                        state    <= TX_LEN_L;
                    end
                end

                TX_LEN_L: begin
                    if (!tx_busy && !tx_start) begin
                        tx_data  <= resp_len[7:0];
                        tx_start <= 1'b1;
                        tx_crc   <= crc16_byte(tx_crc, resp_len[7:0]);
                        state    <= TX_LEN_H;
                    end
                end

                TX_LEN_H: begin
                    if (!tx_busy && !tx_start) begin
                        tx_data  <= resp_len[15:8];
                        tx_start <= 1'b1;
                        tx_crc   <= crc16_byte(tx_crc, resp_len[15:8]);
                        if (resp_len == 16'd0)
                            state <= TX_CRC_L;
                        else
                            state <= TX_PAYLOAD;
                    end
                end

                TX_PAYLOAD: begin
                    if (!tx_busy && !tx_start) begin
                        tx_data  <= resp_buf[resp_cnt];
                        tx_start <= 1'b1;
                        tx_crc   <= crc16_byte(tx_crc, resp_buf[resp_cnt]);
                        resp_cnt <= resp_cnt + 16'd1;
                        if (resp_cnt + 16'd1 == resp_len)
                            state <= TX_CRC_L;
                    end
                end

                TX_CRC_L: begin
                    if (!tx_busy && !tx_start) begin
                        tx_data  <= tx_crc[7:0];
                        tx_start <= 1'b1;
                        state    <= TX_CRC_H;
                    end
                end

                TX_CRC_H: begin
                    if (!tx_busy && !tx_start) begin
                        tx_data  <= tx_crc[15:8];
                        tx_start <= 1'b1;
                        state    <= TX_DONE;
                    end
                end

                TX_DONE: begin
                    if (!tx_busy && !tx_start) begin
                        state <= RX_IDLE;
                    end
                end

                default: state <= RX_IDLE;
            endcase
        end
    end

endmodule
