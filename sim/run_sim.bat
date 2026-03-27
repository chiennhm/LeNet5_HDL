@echo off
REM ============================================================
REM  LeNet-5 Verilog Simulation with Icarus Verilog
REM ============================================================
echo.
echo === LeNet-5 Simulation ===
echo.

REM Move to project root
cd /d "%~dp0\.."

REM Compile
echo [1/2] Compiling...
iverilog -g2005 -o sim/lenet5_sim.vvp ^
    rtl/relu.v ^
    rtl/argmax.v ^
    rtl/conv_layer.v ^
    rtl/maxpool_layer.v ^
    rtl/fc_layer.v ^
    rtl/lenet5_top.v ^
    tb/tb_lenet5.v

if errorlevel 1 (
    echo COMPILATION FAILED!
    pause
    exit /b 1
)

REM Run simulation (from mem/ directory so $readmemh finds hex files)
echo [2/2] Running simulation...
cd mem
vvp ..\sim\lenet5_sim.vvp
cd ..

echo.
echo Waveform saved to: mem\lenet5_wave.vcd
echo Open with GTKWave:  gtkwave mem\lenet5_wave.vcd
echo.
pause
