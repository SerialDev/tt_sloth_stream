import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from client import Client
from protocol import decode_dot_products
from projection import prepare_completion, complete_dot_products, select_completed, classify_winner


def make_forward(head,m,w,model_weights,permutation,plan,sim,client,calls):
 resident_ids=m['token_ids'];rows=m['rows']
 def forward(hidden):
  assert tuple(hidden.shape)==(1,1,1024),'Greedy single-token, single-sequence path only'
  x=hidden[0,0].detach().float().numpy();sx=max(float(np.abs(x).max())/127,float(np.finfo(np.float32).tiny))
  qx=np.clip(np.rint(x/sx),-127,127).astype(np.int64)
  before=sim.tt.cycles if sim else None;begin=time.perf_counter_ns()
  wire_input=qx[permutation];packet=client.dot_products(wire_input);elapsed=(time.perf_counter_ns()-begin)/1e6
  chip_dots=decode_dot_products(packet,m)
  completed=complete_dot_products(chip_dots,wire_input,plan)
  token,score=select_completed(completed,m)
  if sim:assert sim.tt.cycles-before==m['cycles']
  # Independent CPU reference is validation only, not the production result.
  np.testing.assert_array_equal(chip_dots,w@wire_input)
  np.testing.assert_array_equal(completed,model_weights@qx)
  scores=[int(v)*n for v,n in zip(model_weights@qx,m['scale_numerators'])]
  i=max(range(rows),key=lambda j:(scores[j],-j));assert token==resident_ids[i]
  assert score.numerator*m['scale_denominator']==scores[i]*score.denominator
  # These two CPU projections explicitly exclude every resident row.
  logits=torch.full((1,1,head.weight.shape[0]),float('-inf'),dtype=hidden.dtype,device=hidden.device)
  begin=0
  for resident in resident_ids+[head.weight.shape[0]]:
   if begin<resident:logits[:,:,begin:resident]=torch.nn.functional.linear(hidden,head.weight[begin:resident])
   begin=resident+1
  logits[0,0,token]=float(score)*sx
  selected=int(logits[0,0].argmax())
  # Compare with independently computed host-only hybrid greedy decision.
  reference=logits.detach().clone();reference[0,0,resident_ids]=float('-inf')
  reference[0,0,resident_ids[i]]=scores[i]/m['scale_denominator']*sx
  assert int(reference[0,0].argmax())==selected
  without_chip=complete_dot_products([0]*rows,wire_input,plan)
  ablated_token,ablated_score=select_completed(without_chip,m)
  ablated=logits.detach().clone();ablated[0,0,resident_ids]=float('-inf')
  ablated[0,0,ablated_token]=float(ablated_score)*sx
  ablated_selected=int(ablated[0,0].argmax())
  calls.append({'shard_token':token,'global_token':selected,'winner_path':classify_winner(selected,m),'winner_without_chip_contribution':ablated_selected,'chip_contribution_changes_winner':selected!=ablated_selected,'score_numerator':scores[i],'host_round_trip_ms':elapsed,'cycles':m['cycles'] if sim else None})
  return logits
 return forward


def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
 g.add_argument('--sim',action='store_true');g.add_argument('--port')
 p.add_argument('--model',type=Path,default=Path('build/models/Qwen3.5-0.8B'))
 p.add_argument('--prompt',action='append',help='Unrestricted text prompt; may be repeated')
 p.add_argument('--steps',type=int,default=3);p.add_argument('--output',type=Path)
 a=p.parse_args();assert a.steps>0;r=Path(__file__).resolve().parents[1]
 m=json.loads((r/'evidence/weights.json').read_text());w=np.asarray(m['weights'],dtype=np.int64)
 torch.set_num_threads(1)
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.float32).float().eval()
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 eos=model.generation_config.eos_token_id
 stop_ids=set(eos if isinstance(eos,list) else [eos])-{None}
 if tok.eos_token_id is not None:stop_ids.add(tok.eos_token_id)
 resident_ids=m['token_ids'];rows=m['rows'];head=model.lm_head;original=head.forward
 fw=head.weight[resident_ids].detach().numpy();limits=np.array([2**(b-1)-1 for b in m['row_weight_bits']],dtype=np.float32);sc=np.maximum(np.abs(fw).max(axis=1)/limits,np.finfo(np.float32).tiny)
 np.testing.assert_array_equal(sc,np.asarray(m['scales'],dtype=np.float32))
 permutation=np.asarray(m['input_permutation'],dtype=np.int64)
 assert sorted(permutation.tolist())==list(range(1024))
 model_weights=np.clip(np.rint(fw/sc[:,None]),-limits[:,None],limits[:,None]).astype(np.int64)
 np.testing.assert_array_equal(model_weights,np.asarray(m['full_weights_model_order']))
 plan=prepare_completion(m)
 sim=None
 if a.sim:
  from transport_sim import Simulation
  sim=Simulation();stream=sim.stream
 else:
  import serial
  stream=serial.Serial(a.port,115200,timeout=30,write_timeout=30)
 client=Client(stream);calls=[];generations=[]
 forward=make_forward(head,m,w,model_weights,permutation,plan,sim,client,calls)
 try:
  head.forward=forward
  with torch.inference_mode():
   for prompt in (a.prompt or ['The capital of France is','7 + 5 =','Once upon a time']):
    ids=tok(prompt,return_tensors='pt')['input_ids'];generated=[]
    for _ in range(a.steps):
     out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
     token=int(out.logits[0,-1].argmax());generated.append(token)
     if token in stop_ids:break
     ids=torch.cat((ids,torch.tensor([[token]],device=ids.device)),dim=1)
    generations.append({'prompt':prompt,'tokens':generated,'continuation':tok.decode(generated,skip_special_tokens=True),'raw_continuation':tok.decode(generated),'stop_reason':'eos' if generated[-1] in stop_ids else 'length'})
 finally:
  head.forward=original
  if sim:sim.close()
  else:stream.close()
 report={'passed':True,'mode':'simulated board' if a.sim else 'physical board','rtl_sha256':hashlib.sha256((r/'src/project.v').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'resident_coefficients_independently_verified':True,'offloaded_transactions':len(calls),'complete_chip_neuron_winners':sum(c['winner_path']=='complete_chip_neuron' for c in calls),'partially_offloaded_neuron_winners':sum(c['winner_path']=='chip_slice_plus_host_complement' for c in calls),'chip_contribution_changes_winner_count':sum(c['chip_contribution_changes_winner'] for c in calls),'host_complement_coefficients':plan['host_coefficient_count'],'generations':generations,'calls':calls,'scope':'Greedy hybrid inference: resident rows use declared mixed weight precision and INT8 activations and exact local scale selection; nonresident rows use host FP32. Chip returns one complete dot product and three partial sums. Production host computation uses only the 2688 complement coefficients for the partial rows; the complete row has no host contribution. Completed sums match the original four-row integer reference. Winner provenance distinguishes complete-chip, mixed and nonresident-host rows. Chip-removal comparisons use the same hidden state, not an independently generated sequence. Not a float-model equivalence, full-head offload, sampling or acceleration claim. CPU validation overhead included in execution.'}
 output=a.output or r/('evidence/live_sim.json' if a.sim else 'build/live_board.json');output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
