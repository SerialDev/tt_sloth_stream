"""Reproducible board acceptance demo: --sim or --port /dev/...."""
import argparse
import hashlib
import json
import statistics
import platform
import time
from pathlib import Path
import numpy as np
from client import Client
from device import IDENTITY
from protocol import decode_dot_products
from projection import prepare_completion, complete_dot_products, select_completed
from wire import frame, parse, read_line, write_all


def build_report(r,m,plan,directed,records,faults,simulation):
    timings=[v['round_trip_ms'] for v in records]
    report={'passed':True,'mode':'simulated byte stream + board service + public RTL pins' if simulation else 'physical board USB + manual GPIO clocks',
            'host_platform':platform.platform(),'python_version':platform.python_version(),
            'fixture_sha256':hashlib.sha256((r/'evidence/demo_vectors.json').read_bytes()).hexdigest(),
            'rtl_sha256':hashlib.sha256((r/'src/project.v').read_bytes()).hexdigest(),
            'source_sha256':{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [*sorted((r/'host').glob('*.py')),r/'test/bridge.cpp',r/'evidence/weights.json']},
            'full_neuron_token_id':m['full_token_id'],'host_complement_coefficients':plan['host_coefficient_count'],
            'result_semantics':m['result_semantics'],
            'directed_winner_token_ids':directed,'transactions':len(records)+len(faults),'fault_recovery':faults,
            'median_round_trip_ms':statistics.median(timings),'p95_round_trip_ms':float(np.percentile(timings,95)),
            'request_bytes':len(frame(b'W',bytes(1024))),'response_bytes':len(frame(b'R',bytes(12))),
            'records':records,'limitation':'Simulation wall time is not silicon or USB performance. Board mode uses manual clocks, not a 20 MHz streaming engine.'}
    return report


def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--sim',action='store_true');mode.add_argument('--port')
    p.add_argument('--random',type=int,default=100)
    p.add_argument('--row-coverage',action='store_true',help='Force every resident row to win through the board interface')
    p.add_argument('--output',type=Path)
    a=p.parse_args();r=Path(__file__).resolve().parents[1]
    assert IDENTITY == b'SLOTH1:' + hashlib.sha256((r/'src/project.v').read_bytes()).hexdigest().encode(), 'Update firmware identity for changed RTL'
    m=json.loads((r/'evidence/weights.json').read_text())
    w=np.asarray(m['weights'],dtype=np.int64)
    plan=prepare_completion(m)
    full=np.asarray(m['full_weights_model_order'],dtype=np.int64)[:,plan['permutation']]
    simulation=None
    if a.sim:
        from transport_sim import Simulation
        simulation=Simulation();stream=simulation.stream
    else:
        import serial
        stream=serial.Serial(a.port,115200,timeout=30,write_timeout=30)
    records=[];faults=[]
    try:
        client=Client(stream)
        rng=np.random.default_rng(941)
        vectors=[('zero/tie',np.zeros(1024,dtype=np.int64)),('negative extreme',np.full(1024,-128)),('positive extreme',np.full(1024,127))]
        vectors += [('random %d'%i,x) for i,x in enumerate(rng.integers(-128,128,(a.random,1024)))]
        directed=[]
        if a.row_coverage:
            for target,row in enumerate(full):
                x=np.rint(row/np.max(np.abs(row))*127).astype(np.int64)
                expected=[int(v)*scale for v,scale in zip(full@x,m['scale_numerators'])]
                assert max(range(m['rows']),key=lambda j:(expected[j],-j))==target
                vectors.append(('directed resident '+str(m['token_ids'][target]),x))
                directed.append(m['token_ids'][target])
        # Replay the independently captured real Qwen vectors used by validate_qwen.
        qpath=r/'evidence/demo_vectors.json'
        if qpath.exists():
            vectors += [(v['prompt'],np.asarray(v['activations'],dtype=np.int64)) for v in json.loads(qpath.read_text())]
        for label,x in vectors:
            scores=[int(v)*s for v,s in zip(full@x,m['scale_numerators'])]
            i=max(range(m['rows']),key=lambda j:(scores[j],-j))
            cycles=simulation.tt.cycles if simulation else None
            start=time.perf_counter_ns();packet=client.dot_products(x);elapsed=time.perf_counter_ns()-start
            np.testing.assert_array_equal(decode_dot_products(packet,m),w@x)
            completed=complete_dot_products(decode_dot_products(packet,m),x,plan)
            np.testing.assert_array_equal(completed,full@x)
            token,score=select_completed(completed,m)
            assert token==m['token_ids'][i]
            assert score.numerator*m['scale_denominator']==scores[i]*score.denominator
            if simulation:assert simulation.tt.cycles-cycles==m['cycles']
            records.append({'case':label,'token':token,'score_numerator':scores[i],'round_trip_ms':elapsed/1e6,'cycles':m['cycles'] if simulation else None})
        # Each failed request must recover to a known-answer transaction.
        for label,bad in [('CRC',b'W:00:0000\n'),('short payload',frame(b'W',b'\0')),('unknown command',frame(b'Z')),('oversized line',b'x'*2200+b'\n')]:
            write_all(stream,bad)
            assert parse(read_line(stream))[0]==b'E'
            packet=client.dot_products([0]*1024)
            assert decode_dot_products(packet,m)==[0,0,0,0]
            faults.append(label)
    finally:
        if simulation:simulation.close()
        else:stream.close()
    report=build_report(r,m,plan,directed,records,faults,simulation)
    output=a.output or r/('evidence/demo_sim.json' if a.sim else 'build/demo_board.json')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','source_sha256']},indent=2))

if __name__=='__main__':main()
