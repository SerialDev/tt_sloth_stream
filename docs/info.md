## How it works

Thirty-two complete, fixed Qwen3.5-0.8B embedding/LM-head rows consume one
common 1024-element signed INT8 activation vector. The resident token IDs are
11, 25, 198, 220, 264, 271, 279, 328, 393, 421, 907, 944, 2438, 2943, 2973,
6105, 7172, 7806, 8981, 10663, 11748, 11751, 19241, 20620, 21047, 23327, 25358,
30994, 31106, 44299, 55918 and 77916. Rows 25, 198 and 271 keep INT8 weights;
the other 29 use symmetric INT6 weights. The circuit computes 32 dot products,
applies exact rational representations of the per-row FP32 scales, and returns
the highest score and its token ID. Exact ties select the lower token ID.
The weights are standard-cell logic, not SRAM. This is an experimental
inference shard, not a complete LLM.

At 20 MHz a transaction takes 1293 clocks, excluding host and board overhead.
The footprint is 8x4 tiles. Signoff uses reviewed 1.5 ns slew / 0.3 pF global
capacitance limits; stock 0.75 ns / 0.2 pF signoff is not claimed. Library pin
limits still apply.

## How to test

Use the Tiny Tapeout board SDK in ASIC_RP_CONTROL mode, select
`tt_um_serialdev_sloth_stream`, stop auto-clocking, and assert active-low reset.
The RP drives only bidirectional bits 0–2 (`uio_oe_pico = 0x07`).

Drive each signed byte on ui_in and valid on uio[0]. A byte is accepted on a
rising clock edge when ready uio[3] is high. After 1024 bytes, wait for output
valid uio[4]. Read uo_out before each rising edge with output-ready uio[1].
The 12-byte little-endian response contains a signed 64-bit score numerator
followed by an unsigned 32-bit token ID. The score denominator is 17179869184;
multiply by the host activation scale to estimate the logit. A zero vector
must return score 0 and token 11. Use nonzero known-answer vectors as well.

Abort uio[2] clears a transaction on the next enabled rising edge. Reset is
asynchronous. ena low freezes state and disables bidirectional outputs.
Never drive uio[3:7] from the board. Pins and data must meet setup/hold timing.

See host/demo.py and host/live_demo.py for the PC USB demo. The same client
and board service have been exercised against Verilated public pins. Manual
GPIO clocking establishes correctness, not a 20 MHz board data rate.

## External hardware

A compatible Tiny Tapeout SKY130 development board with its supported SDK
and a USB-connected PC. No external PMOD is required for the manual-clock
test. Shuttle allocation and physical board testing remain outstanding.
