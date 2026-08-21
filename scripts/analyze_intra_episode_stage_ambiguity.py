#!/usr/bin/env python3
"""Intra-episode, GT-stage temporal ambiguity search; no cross-episode pairs."""
from __future__ import annotations
import json
from pathlib import Path
import h5py,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED';V2=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/stage_action_ambiguity_v2';V3=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/stage_action_ambiguity_v3';OUT=V3/'intra_episode_stage_ambiguity';EPS=(.5,1.);WINDOW=5
def ft(n,b=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf'%('-Bold' if b else ''),n)
def main():
 OUT.mkdir(parents=True,exist_ok=True);m=json.loads((DATA/'split_manifest.json').read_text());n=json.loads((V2/'normalization_stats_used.json').read_text());sm=np.array(n['physical_mean']);ss=np.array(n['physical_std']);am=np.array(n['action_mean']);ast=np.array(n['action_std']);threshold=json.loads((V3/'transition_aware_ambiguity_results.json').read_text())['threshold_value']
 pairs=[];counts={f'{i}-{i+1}':0 for i in range(4)};per_episode={};representative=None;normal=0
 for eid in m['splits']['train']:
  with h5py.File(m['episode_paths'][eid],'r') as f:
   if str(f.attrs['episode_type'])!='NORMAL_SUCCESS':continue
   if 'active_phase' not in f:raise RuntimeError(f'{eid}: missing active_phase')
   normal+=1;raw=f['full_physical_state'][:].astype('f4');action=f['executed_action'][:].astype('f4');z=f['active_phase'][:].astype(int);state=(raw-sm)/ss;act=(action-am)/ast;found=0
   for i in range(4):
    j=i+1;ti=np.flatnonzero(z==i);tj=np.flatnonzero(z==j)
    if not len(ti) or not len(tj):continue
    d=cdist(state[tj],state[ti]);d[np.abs(tj[:,None]-ti[None,:])<=WINDOW]=np.inf;nn=d.argmin(1);do=d[np.arange(len(tj)),nn];da=np.linalg.norm(act[tj]-act[ti[nn]],axis=1);keep=np.flatnonzero((do<EPS[1])&(da>threshold))
    for q in keep:
     row={'episode_id':str(eid),'stage_transport':i,'stage_place':j,'timestep_i':int(ti[nn[q]]),'timestep_j':int(tj[q]),'temporal_separation':int(abs(tj[q]-ti[nn[q]])),'D_O':float(do[q]),'D_A':float(da[q]),'state_i':raw[ti[nn[q]]].tolist(),'state_j':raw[tj[q]].tolist(),'action_i':act[ti[nn[q]]].tolist(),'action_j':act[tj[q]].tolist(),'epsilon_min':.5 if do[q]<.5 else 1.0}
     pairs.append(row);counts[f'{i}-{j}']+=1;found+=1
     if representative is None or row['D_A']>representative['D_A']:representative={**row,'stage_sequence':z.tolist(),'episode_length':len(z)}
   if found:per_episode[str(eid)]=found
 if normal!=800:raise RuntimeError(f'Expected 800 NORMAL_SUCCESS episodes, found {normal}')
 summary={'DATA_CHECK':'PASS','dataset':'NORMAL_SUCCESS only','stage_source':'GROUND_TRUTH_DATASET_ANNOTATION','stage_field':'active_phase','normalization':'frozen C1/C2 train split','INTRA_EPISODE_ONLY':'YES','CROSS_EPISODE_USED':'NO','TRANSITION_BOUNDARY_REMOVED':'YES','transition_window':WINDOW,'epsilon':list(EPS),'action_threshold_p95_same_stage':threshold,'pair_count':len(pairs),'stage_pair_count':counts,'mean_D_O':None if not pairs else float(np.mean([p['D_O'] for p in pairs])),'median_D_O':None if not pairs else float(np.median([p['D_O'] for p in pairs])),'mean_D_A':None if not pairs else float(np.mean([p['D_A'] for p in pairs])),'median_D_A':None if not pairs else float(np.median([p['D_A'] for p in pairs])),'max_D_A_pair':representative,'per_episode_ambiguity_count':per_episode,'AMBIGUITY_FOUND':'YES' if pairs else 'NO','ANALYSIS_COMPLETE':'YES'}
 (OUT/'intra_episode_ambiguity_pairs.json').write_text(json.dumps(pairs,indent=2)+'\n');(OUT/'intra_episode_ambiguity_report.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Count figure is always factual (including an all-zero outcome), never a synthetic result.
 W,H=1400,720;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);d.text((60,35),'Intra-episode ambiguity count by adjacent stage pair',font=ft(34,True),fill='#17212B');base=590;mx=max(max(counts.values()),1)
 for k,(key,val) in enumerate(counts.items()):
  x=145+k*305;h=val/mx*350;d.rectangle((x,base-h,x+210,base),fill='#C83E3A' if val else '#D5DBE1');d.text((x+25,base+22),f'Stage {key}',font=ft(24,True),fill='#17212B');d.text((x+55,base-h-42),f'n={val}',font=ft(23,True),fill='#17212B')
 d.text((145,660),f'Real same-episode nearest pairs; |Δt|>{WINDOW}, D_O<1.0, D_A>{threshold:.2f} (same-stage P95).',font=ft(20),fill='#59636E');im.save(OUT/'FIG_INTRA_EPISODE_AMBIGUITY_COUNT.png',dpi=(300,300))
 if pairs:
  r=representative;z=np.array(r['stage_sequence']);T=len(z);W,H=1800,720;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);d.text((60,35),'Intra-episode temporal stage ambiguity',font=ft(39,True),fill='#17212B');x0,x1=150,1660;colors=['#7597B7','#A676BA','#3B82B6','#D65F5F','#7B8794'];names=['0 Approach','1 Grasp','2 Transport','3 Place','4 Retreat']
  for s in range(5):
   ix=np.flatnonzero(z==s)
   if len(ix):
    a=x0+ix.min()/max(T-1,1)*(x1-x0);b=x0+ix.max()/max(T-1,1)*(x1-x0);y=190+s*58;d.rectangle((a,y,b+3,y+36),fill=colors[s]);d.text((x0-125,y+4),names[s],font=ft(20,True),fill='#27313B')
  xi=x0+r['timestep_i']/max(T-1,1)*(x1-x0);xj=x0+r['timestep_j']/max(T-1,1)*(x1-x0);yi=190+r['stage_transport']*58+18;yj=190+r['stage_place']*58+18;d.line((xi,yi,xj,yj),fill='#59636E',width=3);d.ellipse((xi-11,yi-11,xi+11,yi+11),fill='#3B82B6');d.ellipse((xj-11,yj-11,xj+11,yj+11),fill='#C83E3A');d.rounded_rectangle((150,570,1660,690),radius=10,fill='#F4F6F8',outline='#D3DAE1');d.text((180,590),f'Same episode: {r["episode_id"]}; Δt={r["temporal_separation"]}, D_O={r["D_O"]:.2f}, D_A={r["D_A"]:.2f}',font=ft(27,True),fill='#17212B');d.text((180,635),'Blue: earlier-stage state; red: later-stage state; dashed line: same-episode ambiguity pair.',font=ft(21),fill='#59636E');im.save(OUT/'FIG_INTRA_EPISODE_TEMPORAL_AMBIGUITY.png',dpi=(300,300))
  s1=np.array(r['state_i']);s2=np.array(r['state_j']);a1=np.array(r['action_i']);a2=np.array(r['action_j']);W,H=1800,980;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);d.text((60,35),'Intra-episode expert-action conflict',font=ft(39,True),fill='#17212B')
  for x,title,s,color in [(110,f'Stage {r["stage_transport"]} state',s1,'#3B82B6'),(960,f'Stage {r["stage_place"]} state',s2,'#C83E3A')]:
   d.rounded_rectangle((x,125,x+730,490),radius=14,fill='#F7F9FB',outline='#D1D8E0',width=2);d.text((x+25,150),title,font=ft(28,True),fill='#17212B');d.text((x+25,220),f'EE position [14:17]: {np.array2string(s[14:17],precision=3)}',font=ft(23),fill='#27313B');d.text((x+25,278),f'Object position [22:25]: {np.array2string(s[22:25],precision=3)}',font=ft(23),fill='#27313B');d.text((x+25,336),f'Gripper/grasp flag [42]: {s[42]:.0f}',font=ft(23),fill='#27313B');d.text((x+25,405),f"timestep: {r['timestep_i'] if x==110 else r['timestep_j']}",font=ft(20),fill=color)
  d.text((80,545),f'D_O={r["D_O"]:.2f}; D_A={r["D_A"]:.2f}; automatic threshold={threshold:.2f}',font=ft(28,True),fill='#17212B');base=785;d.line((100,base,1700,base),fill='#27313B',width=2);mx=max(np.abs(np.r_[a1,a2]).max(),1e-6)
  for i,label in enumerate(['dx','dy','dz','drx','dry','drz','grip']):
   x=190+i*220;h1=a1[i]/mx*165;h2=a2[i]/mx*165;d.rectangle((x,min(base,base-h1),x+52,max(base,base-h1)),fill='#3B82B6');d.rectangle((x+63,min(base,base-h2),x+115,max(base,base-h2)),fill='#C83E3A');d.text((x,805),label,font=ft(20,True),fill='#27313B')
  d.text((125,600),'Normalized expert action: blue = earlier stage, red = later stage',font=ft(22),fill='#59636E');im.save(OUT/'FIG_INTRA_EPISODE_ACTION_CONFLICT.png',dpi=(300,300))
 conclusion='Within the same long-horizon execution, the robot revisits similar physical states at different stages, while expert actions diverge due to different control objectives.' if pairs else 'Intra-episode ambiguity is not observed under the current threshold; cross-episode analysis remains the primary evidence.'
 (OUT/'INTRA_EPISODE_TEMPORAL_AMBIGUITY_REPORT.md').write_text('# Intra-Episode Temporal Stage Ambiguity\n\n%s\n\n- Ambiguity found: **%s**\n- Intra-episode only: **YES**\n- Cross-episode used: **NO**\n- Transition boundary removed: **YES**\n'%(conclusion,summary['AMBIGUITY_FOUND']))
 print(json.dumps({'DATA_CHECK':'PASS','INTRA_EPISODE_ONLY':'YES','CROSS_EPISODE_USED':'NO','TRANSITION_BOUNDARY_REMOVED':'YES','AMBIGUITY_FOUND':summary['AMBIGUITY_FOUND'],'ANALYSIS_COMPLETE':'YES'},indent=2))
if __name__=='__main__':main()
