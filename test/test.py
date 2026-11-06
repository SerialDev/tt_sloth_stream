import json,random
from pathlib import Path
import cocotb
from cocotb.triggers import Timer

@cocotb.test()
async def exact_shard(d):
    root=Path(__file__).resolve().parents[1]
    m=json.loads((root/'evidence/weights.json').read_text())
    d.clk.value=0;d.ena.value=1;d.rst_n.value=0;d.ui_in.value=0;d.uio_in.value=0
    async def tick(value=0,ctrl=0):
        d.ui_in.value=int(value)&255;d.uio_in.value=ctrl
        d.clk.value=0;await Timer(25,unit='ns')
        d.clk.value=1;await Timer(25,unit='ns')
        d.clk.value=0;await Timer(1,unit='ns')
    async def compute(x):
        wire=x
        if m.get('input_encoding')=='bitplanes_msb_first_lsb_packed':
            wire=[sum(((x[base+j]>>bit)&1)<<j for j in range(8))
                  for bit in range(7,-1,-1) for base in range(0,1024,8)]
        for column,value in enumerate(wire):
            for _ in range(40):
                if int(d.uio_out.value)&8:break
                await tick()
            else:raise AssertionError('Input timeout')
            if column==511:
                d.ena.value=0;await tick(99,7);assert int(d.uio_oe.value)==0
                d.ena.value=1
            await tick(value,1)
        for _ in range(10000):
            if int(d.uio_out.value)&16:break
            await tick()
        else:raise AssertionError('Output timeout')
        dots=[sum(a*b for a,b in zip(row,x)) for row in m['weights']]
        scores=[dot*scale for dot,scale in zip(dots,m['scale_numerators'])]
        winner=max(range(m['rows']),key=lambda i:(scores[i],-m['token_ids'][i]))
        packet=bytearray()
        for _ in range(12):
            assert int(d.uio_out.value)&16
            value=int(d.uo_out.value);await tick();assert int(d.uo_out.value)==value
            packet.append(value);await tick(ctrl=2)
        if m.get('result_format')=='four_signed_int24':
            assert [int.from_bytes(packet[3*i:3*i+3],'little',signed=True) for i in range(4)]==dots
        else:
            assert int.from_bytes(packet[:8],'little',signed=True)==scores[winner]
            assert int.from_bytes(packet[8:],'little')==m['token_ids'][winner]
        assert int(d.uio_out.value)&8
    await tick();d.rst_n.value=1;await tick()
    rng=random.Random(20261104)
    vectors=[[0]*1024,[-128]*1024,[127]*1024,[rng.randrange(-128,128) for _ in range(1024)]]
    for target,row in enumerate(m['weights']):
        peak=max(abs(w) for w in row);x=[round(w/peak*127) for w in row]
        scores=[sum(a*b for a,b in zip(weights,x))*scale for weights,scale in zip(m['weights'],m['scale_numerators'])]
        assert max(range(m['rows']),key=lambda i:(scores[i],-m['token_ids'][i]))==target
        vectors += [x,[127 if w>=0 else -128 for w in row],[-128 if w>=0 else 127 for w in row]]
    vectors += [v['activations'] for v in json.loads((root/'evidence/demo_vectors.json').read_text())[:4]]
    for vector in vectors:await compute(vector)
    await tick(42,1);await tick(ctrl=4);await compute(vectors[0])
    await tick(42,1);d.rst_n.value=0;await tick();d.rst_n.value=1;await compute(vectors[-1])
    d._log.info('GATE_TRANSACTIONS=%d',len(vectors)+2)
