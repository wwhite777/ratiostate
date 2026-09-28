#!/usr/bin/env python3
"""Replay saved conditional scores, packed timing, and companion coupling analysis."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, statistics, subprocess, sys, traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from score_lineage import score  # exact frozen scorer source, including numerical implementation
ARMS=('native','bf16_boundary','fp16_boundary','bf16_pair_boundary')
TIMED=('native','bf16_boundary','bf16_pair_boundary')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v): p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def need(ok,msg):
    if not ok: raise ValueError(msg)
def package_hash():
    m=json.loads((ROOT/'package_manifest.json').read_text())
    need(m.get('status')=='private_candidate' and m.get('scope')=='authored-derived saved-record replay for Tables 1-2, packed timing, and Section 5.4','scope mismatch')
    for name,record in m['files'].items():
        path=(ROOT/name).resolve()
        need(path.is_relative_to(ROOT) and path.is_file() and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],f'package input hash mismatch: {name}')
    return sha(ROOT/'package_manifest.json')
def score_rows():
    expected={(r['image'],r['arm']):r for r in json.loads((ROOT/'expected_scores.json').read_text())['rows']}
    need(len(expected)==16,'expected score coverage mismatch')
    rows=[]
    for image in range(4):
        d=json.loads((ROOT/f'targets/image_{image:05d}.json').read_text())
        targets=np.asarray(d['targets'],dtype=np.int64)
        need(len(targets)==783 and 0<=d['conditionedPixel']<=255,'target shape mismatch')
        for arm in ARMS:
            path=ROOT/f'predictions/image_{image:05d}_{arm}.f32'
            params=np.fromfile(path,dtype='<f4')
            need(params.size==783*30,'prediction shape mismatch')
            values=score(params.reshape(783,30),targets)
            bits=float(-np.sum(values['token_log_probability'])/(math.log(2.0)*783))
            old=expected[(image,arm)]
            need(math.isclose(bits,old['bits'],rel_tol=2e-13,abs_tol=2e-13),f'expected score mismatch: {image} {arm}')
            rows.append({'image':image,'label':old['label'],'arm':arm,'bits':bits})
    need(len(rows)==16,'score row count mismatch')
    return rows
def tabulate(rows):
    lookup={(r['image'],r['arm']):r['bits'] for r in rows}
    table1=[]; table2=[]
    for image in range(4):
        x={'image':image,'label':next(r['label'] for r in rows if r['image']==image)}
        x.update({arm:lookup[(image,arm)] for arm in ARMS})
        table1.append(x)
        table2.append({'image':image,'label':x['label'],**{arm:x[arm]-x['native'] for arm in ARMS[1:]}})
    means={arm:float(np.mean([r[arm] for r in table1])) for arm in ARMS}
    mean_deltas={arm:means[arm]-means['native'] for arm in ARMS[1:]}
    return table1,table2,means,mean_deltas
def markdown(table1,table2,means,deltas):
    labels={'native':'FP32','bf16_boundary':'BF16','fp16_boundary':'FP16','bf16_pair_boundary':'Pair'}
    a=['| Index (label) | FP32 | BF16 | FP16 | Pair |','|---|---|---|---|---|']
    for row in table1: a.append('| '+f"{row['image']} ({row['label']})"+' | '+' | '.join(f"{row[x]:.9f}" for x in ARMS)+' |')
    a.append('| Mean | '+' | '.join(f'{means[x]:.9f}' for x in ARMS)+' |')
    b=['| Index (label) | BF16 − FP32 | FP16 − FP32 | Pair − FP32 |','|---|---|---|---|']
    for row in table2: b.append('| '+f"{row['image']} ({row['label']})"+' | '+' | '.join(f"{row[x]:+.9f}" for x in ARMS[1:])+' |')
    b.append('| Mean | '+' | '.join(f'{deltas[x]:+.9f}' for x in ARMS[1:])+' |')
    return '\n'.join(['Table 1. Saved-prediction conditional bits per predicted pixel.','',*a,'','Table 2. Difference from native FP32.','',*b,''])+'\n'
def timing():
    raw=json.loads((ROOT/'timing_raw.json').read_text()); rounds=raw['rounds']
    need(len(rounds)==6 and {tuple(x['order']) for x in rounds}==set(__import__('itertools').permutations(TIMED)),'timing order mismatch')
    payload={}
    for arm in TIMED:
        warm=raw['warm'][arm]
        need(warm['calls']==32,'warm count mismatch')
        record=warm['retained']; payload[arm]=record['bytes']
        need(record['bytes']==sum(record['viewByteLengths']) and record['backingBuffers']==len(record['viewByteLengths']),'retained payload mismatch')
        need(record['bytes']==67584*(2 if arm=='bf16_boundary' else 4),'payload byte count mismatch')
    ratios={arm:[] for arm in TIMED[1:]}
    for index,round_ in enumerate(rounds,1):
        need(round_['round']==index and set(round_['loopSeconds'])==set(TIMED),'timing round schema mismatch')
        base=round_['loopSeconds']['native']; need(math.isfinite(base) and base>0,'invalid timing')
        for arm in TIMED:
            value=round_['loopSeconds'][arm]; need(math.isfinite(value) and value>0,'invalid timing')
            retained=round_['retainedAfterLoop'][arm]
            need(retained['bytes']==payload[arm] and retained['bytes']==sum(retained['viewByteLengths']),'retained round mismatch')
        for arm in ratios: ratios[arm].append(round_['loopSeconds'][arm]/base)
    paired={arm:{'ratios':values,'median':statistics.median(values),'min':min(values),'max':max(values),'screenAtMostFivePercentOverhead':statistics.median(values)<=1.05} for arm,values in ratios.items()}
    old=json.loads((ROOT/'expected_timing.json').read_text())['paired']
    for arm in ratios:
        for key in ('ratios','median','min','max'):
            left,right=paired[arm][key],old[arm][key]
            if isinstance(left,list): need(all(math.isclose(a,b,rel_tol=1e-14,abs_tol=1e-14) for a,b in zip(left,right)) and len(left)==len(right),f'expected timing mismatch: {arm} {key}')
            else: need(math.isclose(left,right,rel_tol=1e-14,abs_tol=1e-14),f'expected timing mismatch: {arm} {key}')
        need(paired[arm]['screenAtMostFivePercentOverhead']==old[arm]['screenAtMostFivePercentOverhead'],'timing screen mismatch')
    fp16=json.loads((ROOT/'fp16_payload_raw.json').read_text())['runs']
    need(len(fp16)==4 and [x['image'] for x in fp16]==list(range(4)),'FP16 payload coverage mismatch')
    for run in fp16:
        need(run['cacheElements']==67584 and run['physicalRetainedPayloadBytesBatch1']==run['cacheElements']*2 and run['backingBuffers']==2 and run['actualFormat']=='Uint16Array IEEE binary16','FP16 physical payload mismatch')
    need(len({run['physicalRetainedPayloadBytesBatch1'] for run in fp16})==1,'FP16 payload differs among images')
    return {'paired':paired,'rawRoundCount':6,'warmCallsPerArm':32,'timedCallsPerRoundPerArm':256,'interpretation':'recorded CPU loop timing only; no new timing, RSS, or peak-memory claim'}, {'stateScalars':67584,'retainedBackingBufferPayloadBytes':{**payload,'fp16_boundary':fp16[0]['physicalRetainedPayloadBytesBatch1']},'scope':'retained boundary payload only; FP16 value from four saved physical Uint16 run records, not packed timing arm'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output).resolve()
    if out.exists() or out.is_relative_to(ROOT): print('refusing existing or in-bundle output directory',file=sys.stderr);return 2
    out.mkdir(parents=True)
    try:
        need(np.__version__=='1.26.4','NumPy version mismatch')
        need(os.getenv('CUDA_VISIBLE_DEVICES')=='','CUDA must be hidden')
        need(all(os.getenv(k)=='2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'numerical thread cap mismatch')
        before=package_hash()
        rows=score_rows(); t1,t2,means,deltas=tabulate(rows)
        dump(out/'scores.json',{'metric':'idealized_sampler_implied_conditional_discretized_nll','rows':rows,'table1':t1,'table2':t2,'means':means,'meanDeltas':deltas,'conditioning':'pixel 0; score pixels 1 through 783'})
        (out/'tables.md').write_text(markdown(t1,t2,means,deltas))
        measured,payload=timing();dump(out/'timing.json',measured);dump(out/'payload.json',payload)
        command=[sys.executable,'-I',str(ROOT/'coupling/replay.py'),'--output',str(out/'coupling')]
        completed=subprocess.run(command,capture_output=True,text=True,env=os.environ.copy(),timeout=600)
        (out/'coupling.stdout.log').write_text(completed.stdout);(out/'coupling.stderr.log').write_text(completed.stderr)
        need(completed.returncode==0,'coupling replay failed')
        need(json.loads((out/'coupling/complete.json').read_text())['status']=='complete','coupling incomplete')
        need(package_hash()==before,'package changed during replay')
        dump(out/'provenance.json',{'packageManifestSha256':before,'isolatedMode':bool(sys.flags.isolated),'pythonVersion':sys.version.split()[0],'numpy':np.__version__,'scoreLineage':'score_lineage.py exact frozen scorer copy','couplingReplay':'coupling/replay.py unchanged v1 companion','scope':'saved-record arithmetic only; no model inference or new timing'})
        files={x.name:{'bytes':x.stat().st_size,'sha256':sha(x)} for x in out.iterdir() if x.is_file()}
        dump(out/'manifest.json',{'files':files,'couplingManifestSha256':sha(out/'coupling/manifest.json')})
        dump(out/'complete.json',{'status':'complete','manifestSha256':sha(out/'manifest.json')})
        print(json.dumps({'status':'complete','scoreRows':len(rows),'couplingRows':24,'timingRounds':6,'output':str(out)}));return 0
    except Exception as e:
        dump(out/'failure.json',{'status':'failed','errorType':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
        print(traceback.format_exc(),file=sys.stderr);return 1
if __name__=='__main__': raise SystemExit(main())
