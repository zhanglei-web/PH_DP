#!/usr/bin/env python3
"""Stage-3-centred temporal ambiguity events from real Normal-success trajectories."""
from __future__ import annotations
import json
from pathlib import Path
import h5py,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED';V2=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/stage_action_ambiguity_v2';V3=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/stage_action_ambiguity_v3';OUT=V3/'stage_centered_temporal_ambiguity';EPS=(.5,1.)
def nearest(q,qe,r,re):
 ds=[];js=[]
 for s in range(0,len(q),32):
  d=cdist(q[s:s+32],r);d[re[None,:]==qe[s:s+32,None]]=np.inf;j=d.argmin(1);ds.append(d[np.arange(len(j)),j]);js.append(j)
 return np.concatenate(ds),np.concatenate(js)
def ft(n,b=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf'%('-Bold' if b else ''),n)
def main():
 OUT.mkdir(parents=True,exist_ok=True);m=json.loads((DATA/'split_manifest.json').read_text());n=json.loads((V2/'normalization_stats_used.json').read_text());sm=np.array(n['physical_mean']);ss=np.array(n['physical_std']);am=np.array(n['action_mean']);ast=np.array(n['action_std']);thr=json.loads((V3/'transition_aware_ambiguity_results.json').read_text())['threshold_value']
 place=[[],[],[],[],[]];transport=[[],[],[],[],[]];normal=0;episodes={}
 for eid in m['splits']['train']:
  with h5py.File(m['episode_paths'][eid],'r') as f:
   if str(f.attrs['episode_type'])!='NORMAL_SUCCESS':continue
   normal+=1;raw=f['full_physical_state'][:].astype('f4');act=f['executed_action'][:].astype('f4');z=f['active_phase'][:].astype(int);episodes[str(eid)]={'state':raw,'action':act,'stage':z}
   state=(raw-sm)/ss;action=(act-am)/ast;t=np.arange(len(z))
   for stage,bucket in [(2,transport),(3,place)]:
    ix=np.flatnonzero(z==stage)
    for dst,v in zip(bucket,[state[ix],action[ix],np.full(len(ix),str(eid)),t[ix],raw[ix]]):dst.append(v)
 if normal!=800:raise RuntimeError(f'Expected 800 NORMAL_SUCCESS episodes, found {normal}')
 ps,pa,pe,pt,praw=(np.concatenate(x) for x in place);ts,ta,te,tt,traw=(np.concatenate(x) for x in transport)
 do,nn=nearest(ps,pe,ts,te);da=np.linalg.norm(pa-ta[nn],axis=1);keep=np.flatnonzero((do<EPS[1])&(da>thr));events=[]
 for q in keep:events.append({'episode_place':str(pe[q]),'timestep_place':int(pt[q]),'episode_transport':str(te[nn[q]]),'timestep_transport':int(tt[nn[q]]),'D_O':float(do[q]),'D_A':float(da[q]),'place_action':pa[q].tolist(),'transport_action':ta[nn[q]].tolist()})
 (OUT/'stage3_centered_ambiguity_events.json').write_text(json.dumps(events,indent=2)+'\n')
 if not events:raise RuntimeError('No Stage3-centred ambiguity event under fixed threshold')
 chosen=max(events,key=lambda x:x['D_A']);ep=episodes[chosen['episode_place']];t=chosen['timestep_place'];stage=ep['stage'];T=len(stage)
 # Figure 1: one actual Place trajectory and its Stage-3 ambiguity event.
 W,H=1800,720;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);d.text((60,35),'Stage-3 centred temporal ambiguity event',font=ft(39,True),fill='#17212B');x0,x1=140,1660;d.text((140,135),f"Place episode: {chosen['episode_place']}  |  matched Transport episode: {chosen['episode_transport']}",font=ft(24),fill='#48525D');colors=['#7597B7','#A676BA','#3B82B6','#D65F5F','#7B8794'];names=['0 Approach','1 Grasp','2 Transport','3 Place','4 Retreat']
 for s in range(5):
  ix=np.flatnonzero(stage==s)
  if len(ix):
   a=x0+ix.min()/max(T-1,1)*(x1-x0);b=x0+ix.max()/max(T-1,1)*(x1-x0);y=250+s*58;d.rectangle((a,y,b+3,y+36),fill=colors[s]);d.text((x0-125,y+4),names[s],font=ft(20,True),fill='#27313B')
 d.line((x0,570,x1,570),fill='#27313B',width=2)
 for q in range(0,T,max(1,T//6)):d.text((x0+q/max(T-1,1)*(x1-x0)-12,582),str(q),font=ft(18),fill='#59636E')
 xe=x0+t/max(T-1,1)*(x1-x0);d.line((xe,215,xe,570),fill='#C83E3A',width=4);d.ellipse((xe-12,188,xe+12,212),fill='#C83E3A');d.text((xe+18,185),'Stage3 ambiguity event',font=ft(23,True),fill='#A52D2A');d.rounded_rectangle((140,620,1660,695),radius=10,fill='#F4F6F8',outline='#D3DAE1');d.text((170,640),f"Place timestep {t} revisits a Transport-like state from another episode: D_O={chosen['D_O']:.2f}, D_A={chosen['D_A']:.2f}",font=ft(25,True),fill='#17212B');im.save(OUT/'FIG_STAGE3_TEMPORAL_AMBIGUITY_TIMELINE.png',dpi=(300,300))
 # Figure 2: direct raw-state and normalized-action comparison for the selected real event.
 re=episodes[chosen['episode_transport']];rp=ep;st=chosen['timestep_transport'];sp=t;left=traw[nn[keep[np.argmax(da[keep])]]];right=rp['state'][sp];at=ta[nn[keep[np.argmax(da[keep])]]];ap=pa[keep[np.argmax(da[keep])]]
 W,H=1800,980;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);d.text((60,35),'Real Transport - Place ambiguity pair',font=ft(39,True),fill='#17212B')
 for x,title,sraw,action,color in [(110,'Transport reference (Stage 2)',left,at,'#3B82B6'),(960,'Place ambiguous state (Stage 3)',right,ap,'#C83E3A')]:
  d.rounded_rectangle((x,130,x+730,520),radius=14,fill='#F7F9FB',outline='#D1D8E0',width=2);d.text((x+25,155),title,font=ft(28,True),fill='#17212B');d.text((x+25,220),f'EE position [14:17]: {np.array2string(sraw[14:17],precision=3)}',font=ft(23),fill='#27313B');d.text((x+25,275),f'Object position [22:25]: {np.array2string(sraw[22:25],precision=3)}',font=ft(23),fill='#27313B');d.text((x+25,330),f'Gripper/grasp flag [42]: {sraw[42]:.0f}',font=ft(23),fill='#27313B');d.text((x+25,390),f'episode / timestep: {chosen["episode_transport"] if x==110 else chosen["episode_place"]} / {st if x==110 else sp}',font=ft(19),fill='#59636E');d.text((x+25,445),'Expert action shown below (normalized 7D)',font=ft(19),fill=color)
 d.text((80,565),f'D_O = {chosen["D_O"]:.2f}     D_A = {chosen["D_A"]:.2f}     Type-A P95 threshold = {thr:.2f}',font=ft(29,True),fill='#17212B');d.line((100,790,1700,790),fill='#27313B',width=2);labels=['dx','dy','dz','drx','dry','drz','grip'];mx=max(np.abs(np.r_[at,ap]).max(),1e-6)
 for i,label in enumerate(labels):
  x=190+i*220;h1=at[i]/mx*170;h2=ap[i]/mx*170;d.rectangle((x,min(790,790-h1),x+52,max(790,790-h1)),fill='#3B82B6');d.rectangle((x+63,min(790,790-h2),x+115,max(790,790-h2)),fill='#C83E3A');d.text((x,810),label,font=ft(20,True),fill='#27313B')
 d.text((125,630),'Normalized expert action',font=ft(25,True),fill='#27313B');d.text((125,660),'blue = Transport; red = Place',font=ft(20),fill='#59636E');im.save(OUT/'FIG_STAGE3_AMBIGUITY_PAIR_COMPARISON.png',dpi=(300,300))
 # Figure 3 is ranking from the all-stage real-pair V3 C2 sweep, retaining its fixed P95 criterion.
 rank=json.loads((V3/'transition_aware_ambiguity_results.json').read_text())['stage_pair_ambiguity_ranking'];rows=[(k,v) for k,v in rank.items() if v['ambiguous_pair_count']];rows.sort(key=lambda q:q[1]['ambiguous_pair_count'],reverse=True);W,H=1500,760;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);d.text((60,35),'Stage-pair ambiguity event count',font=ft(38,True),fill='#17212B');base=640;mx=max(v['ambiguous_pair_count'] for _,v in rows)
 for i,(key,v) in enumerate(rows):
  x=150+i*420;h=v['ambiguous_pair_count']/mx*390;d.rectangle((x,base-h,x+270,base),fill='#C83E3A');d.text((x+50,base+20),f'Stage {key}',font=ft(27,True),fill='#17212B');d.text((x+22,base-h-55),f'n={v["ambiguous_pair_count"]}',font=ft(25,True),fill='#17212B');d.text((x+5,base-h-28),f'mean D_A={v["mean_action_divergence"]:.2f}',font=ft(19),fill='#59636E')
 d.text((150,705),'All bars use real, cross-episode nearest-neighbour pairs with D_O < 1.0 and D_A above the Type-A P95 threshold.',font=ft(19),fill='#59636E');im.save(OUT/'FIG_STAGE_PAIR_AMBIGUITY_COUNT.png',dpi=(300,300))
 report={'DATA_CHECK':'PASS','dataset':'NORMAL_SUCCESS only','stage_source':'GROUND_TRUTH_DATASET_ANNOTATION: active_phase','normal_success_episodes':normal,'query':'every Stage3 Place timestep','stage3_query_count':int(len(ps)),'stage2_reference_count':int(len(ts)),'action_threshold_p95_type_A':thr,'AMBIGUITY_EVENT_COUNT':int(len(events)),'Stage3_finds_Stage2_similar_observations':'YES','States_have_action_divergence':'YES','timeline_position':{'episode_place':chosen['episode_place'],'timestep_place':t,'stage':3},'FINAL_CONCLUSION':'Temporal visualization reveals cross-episode Stage3 states that revisit Stage2-like observations while carrying divergent expert actions. This establishes stage-dependent action ambiguity; it does not by itself establish that ordinary stage transitions are smooth.','FIG_GENERATED':'YES','TEMPORAL_AMBIGUITY_ANALYSIS_COMPLETE':'YES'}
 (OUT/'stage_centered_temporal_ambiguity_summary.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'STAGE_CENTERED_TEMPORAL_AMBIGUITY_REPORT.md').write_text('# Stage-Centered Temporal Ambiguity\n\n- Stage3 finds Stage2 similar observations: **YES**\n- These states have action divergence: **YES**\n- Displayed timeline event: `%s`, timestep %d (Stage3).\n\n%s\n'%(chosen['episode_place'],t,report['FINAL_CONCLUSION']))
 print(json.dumps({'DATA_CHECK':'PASS','AMBIGUITY_EVENT_COUNT':len(events),'FIG_GENERATED':'YES','TEMPORAL_AMBIGUITY_ANALYSIS_COMPLETE':'YES'},indent=2))
if __name__=='__main__':main()
