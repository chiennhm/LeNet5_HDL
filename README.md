# LeNet-5 on FPGA (Altera Cyclone II / DE2 Board)

This repository contains a full hardware implementation of the LeNet-5 Convolutional Neural Network on an FPGA, optimized for resource constraints using external SRAM for weights.

## 1. Architecture Overview
The project implements the standard LeNet-5 architecture:
- **C1**: Conv2d (1 filter → 6 filters, 5x5) + ReLU
- **S2**: AvgPool (2x2)
- **C3**: Conv2d (6 filters → 16 filters, 5x5) + ReLU
- **S4**: AvgPool (2x2)
- **C5**: Conv2d (16 filters → 120 filters, 4x4) + ReLU
- **FC1**: Fully Connected (120 → 84) + ReLU
- **FC2**: Fully Connected (84 → 10)
- **Argmax**: 10 classes to 4-bit digit prediction

### Hardware Optimization
- **Data Representation**: Inputs and intermediate feature maps are stored as 16-bit signed Q8.8 fixed-point numbers. Weights are stored as 8-bit signed Q0.7 (INT8) values. Biases are small internal INT8 register arrays.
- **Memory Hierarchy**: 
  - Intermediate feature maps (`buf_a` and `buf_b`) are stored in internal M4K blocks (Dual-Port RAM).
  - Weights are highly memory-intensive and are streamed from **External SRAM**, preventing the FPGA's internal memory from overflowing.
  - Input images are stored in a dedicated input buffer DPRAM.
- **Math**: Multipliers perform `Q8.8 × INT8(Q0.7) = Q8.15`, accumulated in 32-bit `Q16.15` and scaled back to `Q8.8` with saturation and ReLU.

## 2. Directory Structure
- **`rtl/`**: Complete Verilog source code.
  - `lenet5_top.v`: The top-level Neural Network core, FSM, and memory buffers.
  - `lenet5_de2_top.v`: Board wrapper interfacing the NN core with SRAM, UART, keys, and LEDs.
  - `conv_layer.v`, `fc_layer.v`, `avgpool_layer.v`: Layer-specific compute engines.
- **`tb/`**: Testbenches for verification.
  - `tb_lenet5.v`: Top-level testbench with a built-in simulated SRAM weight model.
- **`sim/`**: Simulation scripts.
  - `run_sim.bat`: Icarus Verilog based compilation and simulation script.
- **`mem/`**: Hex files defining weights, biases, and test images loaded in simulations and Synthesis.
- **`LeNet_py/`**: PyTorch reference model for training, validating, exporting, and quantizing weights.
- **`scripts/`**: Automation scripts for weight generation (`gen_weights.py`), UART image loading (`uart_image_loader.py`), etc.

## 3. Simulation
A standalone `.bat` file is provided to simulate the entire inference process on a predefined test image, leveraging Icarus Verilog:
```bash
cd sim
run_sim.bat
```
This script evaluates the `lenet5_top` over the image found in the `mem/` directory, emulating SRAM delay and capturing complete simulation waveforms into `mem/lenet5_wave.vcd` (viewable via GTKWave).

## 4. Hardware Deployment (DE2 Board)
The Quartus II project `lenet5.qpf` configures the design for the Altera Cyclone II FPGA (EP2C35F672C6).

**Board Interactions:**
- **Clocking:** `CLOCK_50` is used for the synchronous system clock.
- **Reset:** `KEY[0]` is the active-low reset.
- **Control:** `KEY[1]` triggers an inference cycle.
- **External SRAM:** Uses the built-in IS61LV25616AL SRAM for streaming weights during real-time inference.
- **UART:** Uses 115200-8N1 to upload images continuously to the board or monitor statuses/predictions.

## 5. Network Weight Export and Loading
External SRAM requires weights placed sequentially. We follow this absolute memory map layout for generating SRAM dumps:
- **C1**: Address 0
- **C3**: Address 150
- **C5**: Address 2550
- **FC1**: Address 33270
- **FC2**: Address 43350

Weights can be trained using `LeNet_py/train.py`, exported using `LeNet_py/export_weights_quantized.py` to INT8 formats, and packed to the SRAM via UART utilizing `scripts/uart_loader.py` or compiled tightly during board bring-up.