## How it works

This two-tile SKY130 design replaces part of the Qwen3.5-0.8B output projection
with fixed weights implemented in standard-cell logic. It contains one complete
1,024-weight output neuron for token 264 (` a`) and 128-weight slices of the
neurons for tokens 198, 220 and 279. The three slices cover model features 0–127.
All four lanes compute useful contributions for those features; only the complete
neuron contributes for the other 896 features.

The chip retains 1,408 coefficients at the existing INT8/INT6 precisions. Weights
are fixed at fabrication and are not uploaded at runtime. A shared weight decoder
and bitplane arithmetic reduce area. The footprint is `1x2` tiles, with a
161 × 225.76 µm boundary. The top module is `tt_um_serialdev_sloth_stream`.

The host runs the transformer and computes the rest of the output projection.
For each partial neuron, it adds only that neuron's 896 missing terms to the
chip's partial sum. The complete neuron has no host-computed contribution.
The host applies the original per-row scales and merges the results with the
remaining vocabulary. Prompts and generated tokens are unrestricted. The supplied
adapter supports greedy decoding; it does not implement sampling or beam search.

One transaction takes 8,204 unstalled clocks, or 410.2 µs at the 20 MHz timing
target, excluding host and board overhead. The qualified layout uses stock
0.75 ns global slew and 0.2 pF global capacitance constraints, with library pin
limits also enforced. `evidence/qualification.json` records the final layout
hashes, nine-corner timing results and local prechecks for `physical_2tile_eco2`.
That report is the preserved qualification snapshot from before publication.

Local verification passed 2,181 RTL transactions, 22 transactions against the
final powered netlist, 67 board-service transactions and all 15 Tiny Tapeout
prechecks. Both RTL and final netlist checks are functional; extracted static
timing analysis checks timing separately. Forty-four Qwen decoding steps matched
the four-complete-row reference's scores and tokens. These are simulation and
layout results; manufactured-silicon testing remains outstanding.

## How to test

The matching `host/` files and `evidence/weights.json` are required. The metadata
contains the feature permutation, fixed coefficients, host complements and exact
scales. Do not interpret a partial chip sum as a complete output logit.

### Input and output protocol

Reset the design with active-low `rst_n`, then keep `ena` high. Apply the metadata's
`input_permutation` to the 1,024 signed INT8 activations before encoding them.
`host/protocol.py` provides `encode_activations`: it emits planes from bit 7 down
to bit 0, packing eight adjacent permuted features into each byte, with the first
feature in bit 0. The input is still exactly 1,024 bytes.

Present an encoded byte on `ui_in` and assert `uio[0]` (`rx_valid`). It is accepted
on a rising clock edge when `uio[3]` (`rx_ready`) is high. The chip processes the
remaining seven bits during the following seven enabled clocks. Keep clocking
while input-ready is low; only present the next byte when it returns high.

After all input bytes, wait for `uio[4]` (`tx_valid`). Read `uo_out` before each
rising edge with `uio[1]` (`tx_ready`) asserted. The 12-byte response contains
four signed 24-bit little-endian sums in token order `[198, 220, 264, 279]`.
The third sum is complete; the others are partial. A zero vector must return
twelve zero bytes. Use the supplied nonzero fixtures to check the fixed weights.

`uio[2]` aborts on the next enabled rising edge. Reset is asynchronous. `ena` low
freezes state and disables the bidirectional outputs. `uio[5]` reports internal
bit processing; it is not an indication that the entire transaction is idle.
The board must drive only `uio[0:2]`, never `uio[3:7]`.

### Simulated board acceptance

Install Verilator, a C++ compiler and uv. From the repository root, run:

```sh
uv run --no-project --with numpy==2.4.4 python host/demo.py --sim --random 32 --row-coverage --output build/demo_sim.json
uv run --no-project python host/test_transport.py
```

The first command exercises the byte-stream client, board service, public RTL
pins, host complements and all four completed neuron results. The second checks
transport errors, input validation and rejection of a mismatched firmware ID.
The recorded local results remain in `evidence/`.

For unrestricted greedy generation, provide a local Qwen3.5-0.8B checkpoint and
run the matching host adapter. The recorded software versions were NumPy 2.4.4,
PyTorch 2.11.0 and Transformers 5.6.2:

```sh
uv run --no-project --with numpy==2.4.4 --with torch==2.11.0 --with transformers==5.6.2 python host/live_demo.py --sim --model /path/to/Qwen3.5-0.8B --prompt 'Explain in one sentence why ice floats on water. Answer:' --steps 64 --output build/live_sim.json
```

The checkpoint is not included. The adapter verifies the selected model weights
and scales against the metadata. Its CPU reference computations are validation
checks; they do not supply the chip's production contributions.

### Physical board bring-up

Use the compatible Tiny Tapeout board SDK in `ASIC_RP_CONTROL` mode and select
this project after it has been allocated on a shuttle. Stop automatic clocking
and configure `uio_oe_pico = 0x07`. Copy `host/board.py`, `host/device.py` and
`host/wire.py` to the board, then start `device.serve(board.Board(tt))` with the
SDK's configured `tt` object. Use a binary USB stream without a competing REPL.

On the PC, run the same acceptance demo against the board's serial port:

```sh
uv run --no-project --with numpy==2.4.4 --with pyserial python host/demo.py --port /dev/ttyACM0 --random 32 --row-coverage --output build/demo_board.json
```

Replace the port with the board's actual device path. The service's firmware ID
names the RTL revision; nonzero known-answer vectors establish the hardware's
arithmetic identity. This manual-clock procedure checks correctness and does not
establish a 20 MHz board or USB transfer rate.

## External hardware

A compatible Tiny Tapeout SKY130 development board with its supported SDK and
a USB-connected PC. No external PMOD is required for manual-clock bring-up.
The PC runs the rest of Qwen; this chip is a fixed-weight inference proof, not a
standalone language model or a demonstrated end-to-end accelerator.
