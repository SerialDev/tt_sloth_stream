"""Independent functional gate/RTL transaction test; no hierarchy access."""
import json,random
from pathlib import Path
import cocotb
from cocotb.triggers import Timer

@cocotb.test()
async def complete_rows_and_argmax(d):
    m=json.loads((Path(__file__).resolve().parents[1]/'evidence/weights.json').read_text())
    d.clk.value=0;d.ena.value=1;d.rst_n.value=0;d.ui_in.value=0;d.uio_in.value=0
    async def tick(data=0,ctrl=0):
        d.ui_in.value=data&255;d.uio_in.value=ctrl
        d.clk.value=0;await Timer(25,unit='ns')
        d.clk.value=1;await Timer(25,unit='ns')
        d.clk.value=0;await Timer(1,unit='ns')
    await tick();d.rst_n.value=1;await tick()
    rng=random.Random(920)
    vectors=[[0]*1024,[-128]*1024,[127]*1024,[rng.randrange(-128,128) for _ in range(1024)]]
    for target,row in enumerate(m['weights']):
        peak=max(abs(z) for z in row)
        x=[round(z/peak*127) for z in row]
        expected=[sum(a*b for a,b in zip(w,x))*scale for w,scale in zip(m['weights'],m['scale_numerators'])]
        assert max(range(m['rows']),key=lambda i:(expected[i],-i))==target
        vectors.append(x)
    for x in vectors:
        for v in x:
            for wait in range(100):
                if int(d.uio_out.value)&8:break
                await tick()
            else:raise AssertionError('Input timeout')
            await tick(v,1)
        for wait in range(10000):
            if int(d.uio_out.value)&16:break
            await tick()
        else:raise AssertionError('Compute timeout')
        acc=[sum(a*b for a,b in zip(row,x)) for row in m['weights']]
        scores=[v*s for v,s in zip(acc,m['scale_numerators'])]
        winner=max(range(m['rows']),key=lambda i:(scores[i],-i))
        result=bytearray()
        for _ in range(12):
            assert int(d.uio_out.value)==16
            byte=int(d.uo_out.value)
            await tick();assert int(d.uo_out.value)==byte
            result.append(byte);await tick(ctrl=2)
        assert int.from_bytes(result[:8],'little',signed=True)==scores[winner]
        assert int.from_bytes(result[8:],'little')==m.get('token_ids',list(range(m['row_start'],m['row_start']+m['rows'])))[winner]
        assert int(d.uio_out.value)==8
    d._log.info('Verified %d gate/RTL transactions; every resident row selected', len(vectors))
