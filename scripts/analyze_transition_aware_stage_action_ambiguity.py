#!/usr/bin/env python3
"""Transition-aware stage action ambiguity analysis using real Normal-success pairs."""
from __future__ import annotations
import json
from pathlib import Path
import h5py, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED'
V2=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/stage_action_ambiguity_v2'
OUT=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/stage_action_ambiguity_v3'
SEED,CAP,A_CAP=20260820,3000,50000
EPS=(.5,1.)

def nearest(q,qe,r,re):
 ds=[]; js=[]
 for start in range(0,len(q),32):
  d=cdist(q[start:start+32],r); d[re[None,:]==qe[start:start+32,None]]=np.inf; j=d.argmin(1)
  ds.append(d[np.arange(len(j)),j]);js.append(j)
 return np.concatenate(ds),np.concatenate(js)
def ci(x,rng):
 x=np.asarray(x,float); means=np.array([x[rng.integers(len(x),size=len(x))].mean() for _ in range(1000)])
 return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
def stats(state,action,rng):
 return {'count':int(len(state)),'observation_distance':{'mean':float(np.mean(state)),'median':float(np.median(state)),'bootstrap_mean_95_CI':ci(state,rng)},'action_distance':{'mean':float(np.mean(action)),'median':float(np.median(action)),'bootstrap_mean_95_CI':ci(action,rng)}}
def font(n,b=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf'%('-Bold' if b else ''),n)
def axes(d,b,xlabel,ylabel):
 x0,y0,x1,y1=b;d.rectangle(b,outline='#CBD3DC',width=2)
 for z in(.25,.5,.75):
  x=x0+(x1-x0)*z;y=y0+(y1-y0)*z;d.line((x,y0,x,y1),fill='#EEF1F4');d.line((x0,y,x1,y),fill='#EEF1F4')
 d.text(((x0+x1)//2-60,y1+10),xlabel,font=font(23),fill='#4C5662');d.text((x0,y0-32),ylabel,font=font(23),fill='#4C5662')
def ticks(d,b,xlim,ylim):
 x0,y0,x1,y1=b
 for f in(0,.5,1):
  d.text((x0+(x1-x0)*f-12,y1+43),f'{xlim[0]+f*(xlim[1]-xlim[0]):.1f}',font=font(17),fill='#59636E')
  d.text((x0-53,y1-(y1-y0)*f-9),f'{ylim[0]+f*(ylim[1]-ylim[0]):.1f}',font=font(17),fill='#59636E')
def scatter(d,x,y,b,xlim,ylim,color,r):
 keep=(x>=xlim[0])&(x<=xlim[1])&(y>=ylim[0])&(y<=ylim[1]);x=x[keep];y=y[keep];x0,y0,x1,y1=b
 px=x0+(x-xlim[0])/(xlim[1]-xlim[0])*(x1-x0);py=y1-(y-ylim[0])/(ylim[1]-ylim[0])*(y1-y0)
 for a,z in zip(px,py):d.ellipse((a-r,z-r,a+r,z+r),fill=color)
def hist(d,x,b,xlim,color,label,row):
 x0,y0,x1,y1=b;h,e=np.histogram(x,bins=36,range=xlim,density=True);h=h/max(h.max(),1e-8);pts=[]
 for v,l,r in zip(h,e[:-1],e[1:]):pts.append((x0+((l+r)/2-xlim[0])/(xlim[1]-xlim[0])*(x1-x0),y1-v*(y1-y0)))
 d.line(pts,fill=color,width=4,joint='curve');d.text((x0+18,y0+18+row*30),label,font=font(19,True),fill=color)

def main():
 OUT.mkdir(parents=True,exist_ok=True);m=json.loads((DATA/'split_manifest.json').read_text());norm=json.loads((V2/'normalization_stats_used.json').read_text())
 sm=np.asarray(norm['physical_mean']);ss=np.asarray(norm['physical_std']);am=np.asarray(norm['action_mean']);ast=np.asarray(norm['action_std'])
 parts={z:[[],[],[]] for z in range(5)};a_s=[];a_a=[];a_rows=[];b_s=[];b_a=[];b_rows=[];normal=0
 for eid in m['splits']['train']:
  with h5py.File(m['episode_paths'][eid],'r') as f:
   if str(f.attrs['episode_type'])!='NORMAL_SUCCESS':continue
   if 'active_phase' not in f:raise RuntimeError(f'{eid}: active_phase missing')
   normal+=1;x=(f['full_physical_state'][:].astype('f4')-sm)/ss;a=(f['executed_action'][:].astype('f4')-am)/ast;z=f['active_phase'][:].astype(int)
   for stage in range(5):
    ix=np.flatnonzero(z==stage);parts[stage][0].append(x[ix]);parts[stage][1].append(a[ix]);parts[stage][2].append(np.full(len(ix),eid))
   for k in range(1,6):
    ix=np.flatnonzero(z[:-k]==z[k:]);do=np.linalg.norm(x[ix]-x[ix+k],axis=1);da=np.linalg.norm(a[ix]-a[ix+k],axis=1)
    a_s.append(do);a_a.append(da);a_rows.extend({'episode_id':str(eid),'stage':int(z[t]),'k':k,'D_O':float(o),'D_A':float(v)} for t,o,v in zip(ix,do,da))
   # Real adjacent stage boundaries; use last five pre-boundary and first five post-boundary frames.
   edges=np.flatnonzero((z[:-1]+1)==z[1:])
   for edge in edges:
    i,j=int(z[edge]),int(z[edge+1]);pre=np.arange(max(0,edge-4),edge+1);post=np.arange(edge+1,min(len(z),edge+6));pre=pre[z[pre]==i];post=post[z[post]==j]
    for u in pre:
     for v in post:
      b_s.append(float(np.linalg.norm(x[u]-x[v])));b_a.append(float(np.linalg.norm(a[u]-a[v])));b_rows.append({'episode_id':str(eid),'stage_i':i,'stage_j':j,'frame_i':int(u),'frame_j':int(v),'D_O':b_s[-1],'D_A':b_a[-1]})
 if normal!=800:raise RuntimeError(f'Expected 800 NORMAL_SUCCESS episodes, found {normal}')
 rng=np.random.default_rng(SEED);a_s=np.concatenate(a_s);a_a=np.concatenate(a_a);ix=rng.choice(len(a_s),A_CAP,replace=False);a_s,a_a=a_s[ix],a_a[ix];a_rows=[a_rows[i] for i in ix]
 threshold=float(np.quantile(a_a,.95));sample_rng=np.random.default_rng(SEED);sample={}
 for z in range(5):
  x,a,e=(np.concatenate(q) for q in parts[z]);ix=sample_rng.choice(len(x),min(CAP,len(x)),replace=False);sample[z]=(x[ix],a[ix],e[ix])
 c_s=[];c_a=[];c_rows=[];ranking={}
 for i in range(5):
  for j in range(i+1,5):
   x,a,e=sample[i];y,b,f=sample[j];do,nn=nearest(x,e,y,f);keep=np.flatnonzero((do<EPS[1])&(np.linalg.norm(a-b[nn],axis=1)>threshold));da=np.linalg.norm(a[keep]-b[nn[keep]],axis=1)
   key=f'{i}-{j}';ranking[key]={'stage_i':i,'stage_j':j,'nearest_neighbor_queries':int(len(do)),'D_O_lt_0_5':int(np.sum(do<EPS[0])),'D_O_lt_1_0':int(np.sum(do<EPS[1])),'ambiguous_pair_count':int(len(keep)),'mean_action_divergence':None if not len(keep) else float(da.mean())}
   c_s.append(do[keep]);c_a.append(da)
   c_rows.extend({'episode_i':str(e[q]),'episode_j':str(f[nn[q]]),'stage_i':i,'stage_j':j,'D_O':float(do[q]),'D_A':float(v),'epsilon_min':.5 if do[q]<.5 else 1.0,'action_threshold_p95_type_A':threshold} for q,v in zip(keep,da))
 b_s=np.asarray(b_s);b_a=np.asarray(b_a);c_s=np.concatenate(c_s);c_a=np.concatenate(c_a)
 for name,rows in [('type_a_same_stage_continuous_pairs.jsonl',a_rows),('type_b_adjacent_stage_transition_pairs.jsonl',b_rows),('type_c_cross_stage_ambiguous_pairs.jsonl',c_rows)]:
  with (OUT/name).open('w') as f:
   for row in rows:f.write(json.dumps(row)+'\n')
 srng=np.random.default_rng(SEED+2);smooth='YES' if np.median(b_a)<threshold else 'NO';different='YES' if np.median(c_a)>np.median(b_a) else 'NO';conclusion=('Adjacent stages naturally contain smooth transitions. However, a subset of cross-stage states exhibits similar observations with substantially different expert actions, indicating stage-dependent action ambiguity.' if smooth=='YES' and different=='YES' else 'Under the specified ground-truth boundary-window construction, adjacent-stage transitions do not exhibit small action distances. The requested smooth-transition versus ambiguity separation is therefore not supported by this dataset, although cross-stage observation-close, action-divergent pairs do exist.')
 report={'experiment':'Transition_Aware_Stage_Action_Ambiguity','DATA_CHECK':'PASS','data_source':'NORMAL_SUCCESS episodes only','stage_source':'GROUND_TRUTH_DATASET_ANNOTATION','stage_field':'active_phase','normalization':'frozen C1/C2 train-split normalization','normal_success_episodes':normal,'action_divergence_threshold':'95th percentile of Type A D_A','threshold_value':threshold,'type_A_same_stage_continuous':stats(a_s,a_a,srng),'type_B_adjacent_stage_smooth_transition':stats(b_s,b_a,srng),'type_C_cross_stage_ambiguous':stats(c_s,c_a,srng),'stage_pair_ambiguity_ranking':ranking,'Smooth_transition_exists':smooth,'Ambiguous_cross_stage_states_exist':'YES' if len(c_s) else 'NO','Ambiguous_states_differ_from_smooth_transition':different,'FINAL_CONCLUSION':conclusion,'FIG_GENERATED':'YES','TRANSITION_AWARE_AMBIGUITY_COMPLETE':'YES'}
 (OUT/'transition_aware_ambiguity_results.json').write_text(json.dumps(report,indent=2)+'\n')
 (OUT/'TRANSITION_AWARE_AMBIGUITY_REPORT.md').write_text('# Transition-Aware Stage Action Ambiguity\n\n- Smooth transition exists: **%s**\n- Ambiguous cross-stage states exist: **%s**\n- Ambiguous states differ from smooth transition: **%s**\n\n%s\n'%(report['Smooth_transition_exists'],report['Ambiguous_cross_stage_states_exist'],report['Ambiguous_states_differ_from_smooth_transition'],report['FINAL_CONCLUSION']))
 # Scatter: all marks are real pairs; display thinning is deterministic.
 W,H=1800,1050;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);d.text((60,35),'Transition-aware stage action ambiguity',font=font(39,True),fill='#17212B');box=(150,135,1700,770);axes(d,box,'Observation distance  D_O','Action distance  D_A');xlim=(0,max(1.05,float(np.quantile(np.r_[a_s,b_s,c_s],.995))));ylim=(0,max(1.05,float(np.quantile(np.r_[a_a,b_a,c_a],.995))));ticks(d,box,xlim,ylim)
 show=rng.choice(len(a_s),min(16000,len(a_s)),False);scatter(d,a_s[show],a_a[show],box,xlim,ylim,'#3B82B6',1);show=rng.choice(len(b_s),min(8000,len(b_s)),False);scatter(d,b_s[show],b_a[show],box,xlim,ylim,'#35A66F',2);scatter(d,c_s,c_a,box,xlim,ylim,'#C83E3A',5)
 d.ellipse((155,855,176,876),fill='#3B82B6');d.text((185,849),'Type A: same-stage continuous',font=font(23),fill='#27313B');d.ellipse((660,855,681,876),fill='#35A66F');d.text((690,849),'Type B: adjacent-stage smooth transition',font=font(23),fill='#27313B');d.ellipse((1240,855,1261,876),fill='#C83E3A');d.text((1270,849),'Type C: cross-stage ambiguity',font=font(23),fill='#27313B')
 d.rounded_rectangle((150,920,1650,1020),radius=12,fill='#F4F6F8',outline='#D3DAE1');d.text((180,942),f'Automatic ambiguity threshold: Type A 95th percentile D_A = {threshold:.2f}',font=font(25,True),fill='#17212B');d.text((180,980),f'Type B median D_A = {np.median(b_a):.2f}; Type C median D_A = {np.median(c_a):.2f}',font=font(23),fill='#4C5662');im.save(OUT/'FIG_C2_TRANSITION_AWARE_SCATTER.png',dpi=(300,300))
 # Distribution figure.
 W,H=1800,900;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);d.text((60,35),'Transition-aware distribution comparison',font=font(39,True),fill='#17212B');left,right=(120,145,835,640),(960,145,1675,640);axes(d,left,'Observation distance  D_O','Relative density');axes(d,right,'Action distance  D_A','Relative density');ol=(0,max(1.,float(np.quantile(np.r_[a_s,b_s,c_s],.995))));al=(0,max(1.,float(np.quantile(np.r_[a_a,b_a,c_a],.995))));ticks(d,left,ol,(0,1));ticks(d,right,al,(0,1))
 for n,x,color,row in [('Type A: same-stage',a_s,'#3B82B6',0),('Type B: smooth transition',b_s,'#35A66F',1),('Type C: ambiguity',c_s,'#C83E3A',2)]:hist(d,x,left,ol,color,n,row)
 for n,x,color,row in [('Type A: same-stage',a_a,'#3B82B6',0),('Type B: smooth transition',b_a,'#35A66F',1),('Type C: ambiguity',c_a,'#C83E3A',2)]:hist(d,x,right,al,color,n,row)
 d.rounded_rectangle((120,710,1675,850),radius=12,fill='#F4F6F8',outline='#D3DAE1');d.text((150,735),f'Observation medians: A={np.median(a_s):.2f}, B={np.median(b_s):.2f}, C={np.median(c_s):.2f}',font=font(24,True),fill='#17212B');d.text((150,790),f'Action medians: A={np.median(a_a):.2f}, B={np.median(b_a):.2f}, C={np.median(c_a):.2f}  (C uses D_A > {threshold:.2f})',font=font(24,True),fill='#17212B');im.save(OUT/'FIG_C2_TRANSITION_AWARE_DISTRIBUTIONS.png',dpi=(300,300))
 # Ranking figure (no pre-selected stage pair).
 rows=[(k,v) for k,v in ranking.items() if v['ambiguous_pair_count']];rows.sort(key=lambda q:(q[1]['ambiguous_pair_count'],q[1]['mean_action_divergence']),reverse=True);W,H=1500,800;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);d.text((60,35),'Cross-stage ambiguity ranking',font=font(37,True),fill='#17212B');base=680;maxn=max([v['ambiguous_pair_count'] for _,v in rows]or[1]);barw=260
 for n,(key,v) in enumerate(rows):
  x=140+n*400;h=v['ambiguous_pair_count']/maxn*430;d.rectangle((x,base-h,x+barw,base),fill='#C83E3A');d.text((x+35,base+18),f'Stage {key}',font=font(25,True),fill='#17212B');d.text((x+15,base-h-65),f'n={v["ambiguous_pair_count"]}',font=font(23,True),fill='#17212B');d.text((x+3,base-h-35),f'mean D_A={v["mean_action_divergence"]:.2f}',font=font(19),fill='#4C5662')
 d.text((140,735),'Bar height: number of real cross-episode ambiguous pairs (D_O < 1.0; D_A above Type-A 95th percentile).',font=font(20),fill='#4C5662');im.save(OUT/'FIG_C2_STAGE_PAIR_AMBIGUITY_RANKING.png',dpi=(300,300))
 print(json.dumps({'DATA_CHECK':'PASS','PAIR_COUNTS':{'Type_A':len(a_s),'Type_B':len(b_s),'Type_C':len(c_s)},'FIG_GENERATED':'YES','TRANSITION_AWARE_AMBIGUITY_COMPLETE':'YES'},indent=2))
if __name__=='__main__':main()
