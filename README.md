# LeNet-5 HDL on DE2

This project implements LeNet-5 inference in synthesizable Verilog for the
Altera/Terasic DE2 board. The design is intentionally resource-conscious:
feature maps are stored in internal M4K RAMs, weights are streamed from
external SRAM, and layers execute sequentially under a top-level FSM.

The repository also contains the PyTorch reference model and export scripts
used to train, quantize, and prepare weights for the HDL implementation.

## Repository Layout

```text
rtl/        Verilog RTL modules
tb/         RTL testbench
sim/        Simulation helper script
scripts/    UART loaders, image tools, conversion/check scripts
LeNet_py/   PyTorch model, training, quantization, export scripts
 document/  LaTeX report and figures
```

Note that the report directory is named with a leading space: ` document/`.

## Network Implemented

The hardware follows the PyTorch model in `LeNet_py/model.py`:

| Stage | Operation | Input | Output |
| --- | --- | --- | --- |
| Input | 8-bit grayscale MNIST image | 28x28x1 | 28x28x1 |
| C1 | Conv 5x5, 1 -> 6, ReLU | 28x28x1 | 24x24x6 |
| S2 | AvgPool 2x2 | 24x24x6 | 12x12x6 |
| C3 | Conv 5x5, 6 -> 16, ReLU | 12x12x6 | 8x8x16 |
| S4 | AvgPool 2x2 | 8x8x16 | 4x4x16 |
| C5 | Conv 4x4, 16 -> 120, ReLU | 4x4x16 | 1x1x120 |
| FC1 | Linear 120 -> 84, ReLU | 120 | 84 |
| FC2 | Linear 84 -> 10 | 84 | 10 |
| Argmax | Largest logit index | 10 | digit 0-9 |

There is no softmax in hardware. Argmax is enough for classification because
softmax does not change which logit is largest.

## High-Level Mechanism

The core inference module is `rtl/lenet5_top.v`.

The full system operates as follows:

1. A 28x28 image is written into `buf_input` through the `pixel_*` interface.
2. A one-cycle `start` pulse starts inference.
3. The top-level FSM runs one layer at a time:

   ```text
   ST_IDLE -> C1 -> S2 -> C3 -> S4 -> C5 -> FC1 -> FC2 -> Argmax -> ST_DONE
   ```

4. Each layer reads from one feature-map buffer and writes to another.
5. Convolution and fully-connected layers stream weights from external SRAM.
6. When Argmax finishes, `done` is asserted and `digit_out` holds the result.

Only one layer is active at any time. This keeps control simple and lets the
same buffers be reused across multiple stages.

## Data Format

### Feature Maps

Feature maps use signed 16-bit Q8.8 fixed point.

The input image is 8-bit grayscale. Pixels are written as:

```verilog
{8'b0, pixel_data}
```

This places the pixel byte in the fractional part of Q8.8. A pixel value of
255 therefore represents approximately `255 / 256 = 0.996`.

### Weights and Biases

Weights and biases are 8-bit signed INT8, interpreted by the RTL as Q0.7.

Convolution and fully-connected weights are stored in external SRAM. Biases are
small enough to stay inside each layer module as register arrays loaded by
`$readmemh`.

The MAC format is:

```text
input Q8.8 x weight Q0.7 = product Q8.15
accumulator = 32-bit signed Q16.15
output = acc[22:7] -> Q8.8
```

Before writing results back to feature-map RAM, the RTL applies saturation. For
convolution layers and FC1, it also applies ReLU.

## Feature-Map Buffers

`lenet5_top` instantiates three internal RAM buffers:

| Buffer | Depth | Width | Purpose |
| --- | ---: | ---: | --- |
| `buf_input` | 784 | 16 | Input image |
| `buf_a` | 3456 | 16 | C1, C3, C5, FC2 outputs |
| `buf_b` | 864 | 16 | S2, S4, FC1 outputs |

### Ping-Pong Buffer Mechanism

The architecture uses a "ping-pong" memory mechanism to drastically optimize block RAM (M4K) usage on the FPGA. Instead of allocating a separate memory block for the output of every single layer—which would consume excessive resources—the design reuses just two physical dual-port RAMs: `buf_a` and `buf_b`.

Because the top-level FSM executes only one layer at a time sequentially, the feature map data flows back and forth (like a ping-pong ball) between these two buffers:
- **C1** reads from `buf_input` and writes its output to **`buf_a`**.
- **S2** reads from **`buf_a`** and writes its output to **`buf_b`**.
- **C3** reads from **`buf_b`** and writes its output to **`buf_a`**.
- **S4** reads from **`buf_a`** and writes its output to **`buf_b`**.
- **C5** reads from **`buf_b`** and writes its output to **`buf_a`**.
- **FC1** reads from **`buf_a`** and writes its output to **`buf_b`**.
- **FC2** reads from **`buf_b`** and writes its output to **`buf_a`**.
- **Argmax** reads from **`buf_a`** to determine the final classification.

By doing this, the hardware only needs to size each buffer for the *largest* single tensor it will ever hold during the inference process. For `buf_a`, the largest output is from C1 (24x24x6 = 3456 words). For `buf_b`, the largest output is from S2 (12x12x6 = 864 words).

`rtl/dpram.v` is the internal memory primitive. It has one write port and one
registered read port. This means read data is valid one clock after the address
is presented, so consumers include wait states where needed.

## SRAM Weight Layout

Weights are stored as contiguous INT8 bytes in external SRAM:

| Layer | Weights | Base address | Last address |
| --- | ---: | ---: | ---: |
| C1 | 150 | 0 | 149 |
| C3 | 2400 | 150 | 2549 |
| C5 | 30720 | 2550 | 33269 |
| FC1 | 10080 | 33270 | 43349 |
| FC2 | 840 | 43350 | 44189 |

Total external weight storage is 44,190 bytes.

`lenet5_top` adds each layer's local weight address to the appropriate base
address before sending the request to the SRAM controller.

## Weight Transmission and Reception Process

Because the DE2 board has limited internal memory, all convolution and fully-connected layer weights are stored in external SRAM. The weights are transmitted from the PC to the FPGA using a custom binary UART protocol before inference begins.

The process is divided into two sides: Host (PC) and Device (FPGA).

### 1. Host Side (Python Script)
The script `scripts/uart_loader.py` handles parsing the `.hex` weight files and sending them to the FPGA.
- **Parsing**: It reads the `$readmemh`-format files from the `mem/` directory and converts them into raw binary byte streams.
- **Chunking**: To accommodate UART buffer limits and protocol constraints, the data is divided into chunks (maximum 507 data bytes per payload).
- **Frame Construction**: For each chunk, it builds a `WRITE_SRAM` frame (Command `0x01`):
  `[SOF=0xA5][CMD][SEQ][LEN_L][LEN_H][ADDR_0][ADDR_1][ADDR_2][SIZE_L][SIZE_H][DATA...][CRC_L][CRC_H]`
  - `ADDR_0` to `ADDR_2` specify the 24-bit base address for this chunk.
  - `SIZE_L` and `SIZE_H` specify the number of weight bytes in the chunk.
- The script waits for an acknowledgment from the device before sending the next chunk, ensuring reliable transmission over UART.

### 2. Device Side (FPGA RTL)
The module `rtl/uart_protocol.v` acts as the UART endpoint and SRAM programmer.
- **Receiving & Buffering**: It reads incoming UART bytes, verifying the sequence number and CRC-16 checksum. Valid data bytes are temporarily stored in an internal register array (`payload_buf`).
- **SRAM Writing**: Once a complete `WRITE_SRAM` frame is received and verified, the FSM transitions to `WR_ISSUE` and `WR_WAIT` states. It sequentially fetches bytes from `payload_buf` and sends write requests to `sram_controller_de2.v`.
- **Handshaking**: For every byte, `uart_protocol.v` asserts `sram_wr_req` and waits for `sram_wr_done` from the SRAM controller, properly handling the timing requirements of the physical external SRAM chips.
- **Acknowledgment**: After the entire chunk is successfully written to SRAM, it sends a response frame back to the PC:
  `[SOF=0x5A][SEQ][STATUS=0x00][LEN_L=2][LEN_H=0][SIZE_L][SIZE_H][CRC_L][CRC_H]`
  This tells the host to proceed with sending the next chunk of weights.

### 3. Inference Read
During inference, the compute layers (`conv_layer.v`, `fc_layer.v`) request weights directly from the SRAM controller by asserting their read addresses. The UART interface should be inactive during this time to prevent SRAM read/write access contention.

## Module Mechanisms

### `lenet5_top.v`

`lenet5_top` is the core scheduler and interconnect.

Its responsibilities are:

- Store the input image in `buf_input`.
- Instantiate all compute layers.
- Select buffer read/write ports based on the current top-level state.
- Multiplex weight requests from active layers onto one SRAM read port.
- Generate one-cycle start pulses for each layer.
- Assert top-level `done` when Argmax finishes.

The top-level FSM does not overlap layers. This avoids cross-layer hazards and
makes the memory access pattern deterministic.

### `conv_layer.v`

`conv_layer` is parameterized and reused for C1, C3, and C5.

Parameters:

- `IN_SIZE`
- `IN_CH`
- `OUT_CH`
- `KERNEL`
- `BIAS_FILE`

Internal loop counters:

```verilog
oc, oy, ox, ic, ky, kx
```

They represent:

- output channel
- output row/column
- input channel
- kernel row/column

The layer FSM is:

```text
S_IDLE -> S_BIAS -> S_WAIT -> S_MAC -> S_WRITE -> S_DONE
```

Mechanism:

1. `S_BIAS`: load the output-channel bias into `acc`.
2. Request the first weight from SRAM.
3. `S_WAIT`: wait until `w_valid` is asserted.
4. `S_MAC`: multiply `in_data * w_data_q`, sign-extend, accumulate.
5. Advance `kx`, `ky`, and `ic`.
6. Repeat until all kernel products for one output pixel are accumulated.
7. `S_WRITE`: convert accumulator to Q8.8, saturate, apply ReLU, write output.
8. Advance `ox`, `oy`, and `oc`.

Address generation avoids runtime multipliers. Instead of recomputing:

```text
ic * IN_SIZE^2 + (oy + ky) * IN_SIZE + (ox + kx)
```

the RTL tracks base addresses and updates them using compile-time constants:

- `KY_STEP`
- `IC_STEP`
- `OY_STEP`

This is important for Cyclone II because it reduces logic and avoids wasting
embedded multipliers on address arithmetic.

### `avgpool_layer.v`

`avgpool_layer` implements 2x2 average pooling with stride 2.

The FSM is:

```text
S_IDLE -> S_INIT -> S_WAIT -> S_ACC -> S_WRITE -> S_DONE
```

Mechanism:

1. `S_INIT`: clear `sum` for the current 2x2 window.
2. Present the first input address.
3. `S_WAIT`: wait one cycle for synchronous RAM read data.
4. `S_ACC`: sign-extend and add `in_data` into an 18-bit sum.
5. Repeat for the four elements in the 2x2 window.
6. `S_WRITE`: write `sum[17:2]`, which is equivalent to signed divide by 4.

Addressing is also incremental. The module tracks:

- `ch_base`: base address of the current channel
- `row_base`: base address of the current output row
- `pix_base`: base address of the current pooling window

### `fc_layer.v`

`fc_layer` is reused for FC1 and FC2.

The FSM is:

```text
S_IDLE -> S_BIAS -> S_WAIT -> S_MAC -> S_WRITE -> S_DONE
```

Mechanism:

1. Select output neuron `j`.
2. Load `biases[j]` into `acc`.
3. Start reading weights from SRAM at `w_base`.
4. Read input vector entries sequentially from the input buffer.
5. Accumulate `IN_SIZE` products.
6. Write output neuron `j`.
7. Advance `j` and `w_base`.

`APPLY_RELU` controls activation:

- FC1 sets `APPLY_RELU = 1`.
- FC2 sets `APPLY_RELU = 0`, because final logits must preserve sign.

### `argmax.v`

`argmax` reads 10 signed logits from `buf_a`.

The FSM is:

```text
S_IDLE -> S_WAIT -> S_CMP -> S_OUT
```

Mechanism:

1. Initialize `max_val` to the most negative 16-bit number.
2. Read each logit from RAM.
3. Wait one cycle for registered RAM data.
4. Compare with `max_val`.
5. Save the largest value and its index.
6. Output `class_out = max_idx`.

### `sram_controller_de2.v`

This module converts byte-level read/write requests into DE2 SRAM pin
transactions.

Important details:

- External SRAM is 16-bit wide.
- Weights are addressed as bytes.
- `addr[18:1]` selects the SRAM word.
- `addr[0]` selects lower or upper byte.
- Writes have priority over reads.

Read flow:

```text
S_IDLE -> S_RD_WAIT -> S_RD_CAPTURE -> S_IDLE
```

Write flow:

```text
S_IDLE -> S_WR_SETUP -> S_WR_PULSE -> S_WR_HOLD -> S_WR_DONE -> S_IDLE
```

In normal use, weights are written before inference begins, so write/read
contention should not occur during inference.

### `lenet5_de2_top.v`

This is the DE2 board wrapper.

It connects:

- `CLOCK_50` as system clock
- `KEY[0]` as active-low reset
- `KEY[1]` as manual start
- LEDs and HEX display for result/status
- SRAM pins
- UART RX/TX pins

At reset, it can load a default image from `mem/test_image.hex`. The host can
later overwrite the input image through UART.

### `uart_protocol.v`

The UART protocol allows a PC to:

- check connectivity
- write weights to SRAM
- write input images
- start inference
- read status and predicted digit

Host-to-device frame:

```text
[SOF=0xA5][CMD][SEQ][LEN_L][LEN_H][PAYLOAD...][CRC_L][CRC_H]
```

Device-to-host frame:

```text
[SOF=0x5A][SEQ][STATUS][LEN_L][LEN_H][PAYLOAD...][CRC_L][CRC_H]
```

CRC is CRC-16/CCITT-FALSE.

Commands:

| Command | Name | Purpose |
| ---: | --- | --- |
| `0x00` | `PING` | Check link |
| `0x01` | `WRITE_SRAM` | Write INT8 weights to SRAM |
| `0x02` | `START_INFERENCE` | Start the core |
| `0x03` | `GET_STATUS` | Return `done` and `digit_out` |
| `0x04` | `WRITE_IMAGE` | Write pixels into the input buffer |

## Testbench Mechanism

`tb/tb_lenet5.v` tests the core without the physical DE2 board.

It includes:

- A simulated SRAM model for weights.
- Host-side SRAM write tasks.
- Image loading into `buf_input`.
- A cycle counter active only during inference.
- A watchdog timeout.
- VCD waveform dumping.

Testbench flow:

1. Load weight hex files into arrays.
2. Write weights into the simulated SRAM address map.
3. Load one 28x28 image hex file.
4. Write all 784 pixels into the DUT.
5. Verify the image buffer contents.
6. Pulse `start`.
7. Wait for `done`.
8. Print predicted digit and inference cycles.

Default image macro:

```verilog
`define IMG_HEX_FILE "mem/test_img_61_label_8.hex"
```

You can override it at compile time if your simulator flow defines
`IMG_HEX_FILE`.

## Python and Weight Preparation

The PyTorch model is defined in:

```text
LeNet_py/model.py
```

Training is handled by:

```text
LeNet_py/train.py
```

Quantization and export are handled by:

```text
LeNet_py/export_weights_quantized.py
scripts/convert_q88_to_int8.py
```

The RTL expects INT8 Q0.7 weights/biases in `mem/*.hex`. If you first export
Q8.8 files, use:

```bash
python scripts/convert_q88_to_int8.py
```

Expected memory files include:

```text
mem/conv1_weights.hex
mem/conv1_bias.hex
mem/conv3_weights.hex
mem/conv3_bias.hex
mem/c5_weights.hex
mem/c5_bias.hex
mem/fc1_weights.hex
mem/fc1_bias.hex
mem/fc2_weights.hex
mem/fc2_bias.hex
```

The repository snapshot may not include a generated `mem/` directory. Generate
or copy these files before running RTL simulation or Quartus synthesis.

## Simulation

If Icarus Verilog is installed and the `mem/` files exist:

```bash
iverilog -g2005 -o sim/lenet5_sim.vvp \
    rtl/dpram.v rtl/argmax.v rtl/conv_layer.v \
    rtl/avgpool_layer.v rtl/fc_layer.v \
    rtl/lenet5_top.v tb/tb_lenet5.v

vvp sim/lenet5_sim.vvp
gtkwave lenet5_wave.vcd
```

Expected testbench output format:

```text
Inference complete!
  Predicted digit : <digit>
  Inference cycles: <cycles>
```

## DE2 Runtime Flow

The intended board flow is:

1. Reset the board with `KEY[0]`.
2. Use UART `WRITE_SRAM` to load INT8 weights into external SRAM.
3. Use UART `WRITE_IMAGE` to load a 28x28 image.
4. Use UART `START_INFERENCE`.
5. Poll UART `GET_STATUS`, or watch LEDs/HEX.
6. Read `digit_out` when `done` is high.

`LEDR[3:0]` and `HEX0` show the predicted digit. `LEDR[17]` and `LEDG[0]`
show completion status.

## Important Design Assumptions

- One layer runs at a time.
- Layer start signals are one-cycle pulses.
- Internal RAM reads are synchronous, so consumers use wait states.
- Conv/FC weights are not stored in internal block RAM; they are streamed from
  external SRAM.
- Biases are internal register arrays initialized with `$readmemh`.
- Feature maps are Q8.8.
- Weights and biases are INT8 Q0.7.
- Conv and FC accumulators are 32-bit signed.
- Output conversion uses saturation before ReLU.
- FC2 does not apply ReLU.
- Argmax compares raw signed logits.

## Useful Files

| File | Why it matters |
| --- | --- |
| `rtl/lenet5_top.v` | Main core scheduler and buffer interconnect |
| `rtl/conv_layer.v` | Convolution engine and most important datapath |
| `rtl/fc_layer.v` | Fully-connected engine |
| `rtl/avgpool_layer.v` | Pooling engine |
| `rtl/argmax.v` | Final classifier |
| `rtl/sram_controller_de2.v` | Physical SRAM interface |
| `rtl/uart_protocol.v` | PC-to-FPGA command protocol |
| `rtl/lenet5_de2_top.v` | DE2 integration wrapper |
| `tb/tb_lenet5.v` | End-to-end RTL verification |
| `LeNet_py/model.py` | Software reference architecture |
| `LeNet_py/export_weights_quantized.py` | Weight export |
| `scripts/uart_loader.py` | Host-side UART loader |

## Current Limitations

- The design is sequential and not fully parallelized.
- SRAM read latency directly affects inference cycles.
- The current RTL uses one MAC stream per instantiated compute layer.
- The repository may need generated `mem/*.hex` files before simulation.
- Quartus resource and timing reports are not included in this snapshot.

