#!/usr/bin/env python3
"""Retrospective, no-inference analysis of frozen selected-state and score files."""
import argparse
import csv
import hashlib
import json
import math
import time
from decimal import Decimal
from pathlib import Path

import mpmath as mp
import numpy as np

ARMS=('A00','A10','A01','A11','fp16_boundary','bf16_pair_boundary')
CONTRASTS={'primary_A11_minus_A10':(3,1),'matrix_only_A10_minus_A00':(1,0),
           'normalizer_only_A01_minus_A00':(2,0),'joint_A11_minus_A00':(3,0),
           'interaction':None,'matrix_given_bf16_normalizer':(3,2),
           'A10_minus_fp16':(1,4),'fp16_minus_A00':(4,0),
           'pair_minus_A00':(5,0)}
FROZEN_NAMES={'primary_A11_minus_A10':'normalizer_given_bf16_matrix',
              'matrix_only_A10_minus_A00':'matrix_only','normalizer_only_A01_minus_A00':'normalizer_only',
              'joint_A11_minus_A00':'both','interaction':'interaction','A10_minus_fp16':'A10_minus_fp16',
              'matrix_given_bf16_normalizer':'matrix_given_bf16_normalizer',
              'fp16_minus_A00':'fp16_minus_native','pair_minus_A00':'pair_minus_native'}
WINDOWS=((0,128),(128,256),(256,512),(512,783))
FRACTIONS=(.01,.10,.25,.50)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(a):
    a=np.asarray(a,dtype='float64');n=a.size
    mean=float(a.mean());sd=float(a.std(ddof=1));se=sd/math.sqrt(n)
    if sd==0:
        t=p=d=None
    else:
        t=mean/se;mp.mp.dps=50
        p=float(mp.betainc((n-1)/2,.5,0,(n-1)/((n-1)+t*t),regularized=True))
        d=mean/sd
    return {'n':n,'mean':mean,'sample_sd':sd,'standard_error':se,'paired_t_df':n-1,
            'paired_t':t,'two_sided_p_exploratory':p,'paired_standardized_d':d,
            'mean_absolute':float(np.abs(a).mean()),'min':float(a.min()),'max':float(a.max()),
            'positive':int((a>0).sum()),'negative':int((a<0).sum()),'zero':int((a==0).sum())}

def describe(a):
    a=np.asarray(a,dtype='float64')
    return {'n':int(a.size),'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1)),
            'min':float(a.min()),'max':float(a.max())}

def mass(raw,feed):
    raw=np.asarray(raw,dtype='<f4');feed=np.asarray(feed,dtype='<f4')
    assert raw.shape==feed.shape and raw.ndim==3 and raw.shape[1:]==(2,32)
    assert np.isfinite(raw).all() and np.isfinite(feed).all()
    prior=np.concatenate((np.zeros_like(feed[:1]),feed[:-1]),axis=0)
    d=raw.astype('float64')-prior.astype('float64')
    e=feed.astype('float64')-raw.astype('float64')
    H=np.cumsum(d,axis=0,dtype='float64')
    D=np.cumsum(e,axis=0,dtype='float64')
    residual=feed.astype('float64')-H-D
    endpoint_independent=feed[-1].astype('float64')-H[-1]
    assert np.max(np.abs(endpoint_independent-D[-1])) < 1e-8
    assert np.max(np.abs(residual)) < 1e-8
    absorbed=(d>0)&(feed==prior)
    return d,e,H,D,absorbed,float(np.max(np.abs(residual)))

def synthetic_checks():
    raw=np.array([.25,.75,1.25],dtype='<f4').reshape(3,1,1)
    feed=np.array([0,1,1],dtype='<f4').reshape(3,1,1)
    # Pad to the production layer/coordinate shape so the same function is tested.
    raw=np.broadcast_to(raw,(3,2,32)).copy();feed=np.broadcast_to(feed,(3,2,32)).copy()
    d,e,H,D,absorbed,residual=mass(raw,feed)
    assert np.array_equal(d[:,0,0],[.25,.75,.25])
    assert np.array_equal(e[:,0,0],[-.25,.25,-.25])
    assert np.array_equal(H[:,0,0],[.25,1,1.25])
    assert np.array_equal(D[:,0,0],[-.25,0,-.25])
    assert np.array_equal(absorbed[:,0,0],[True,False,True])
    assert sum(float(x) for x in d[:,0,0][absorbed[:,0,0]])==.5
    assert sum(float(x) for x in e[:,0,0][e[:,0,0]>0])==.25
    assert sum(float(-x) for x in e[:,0,0][e[:,0,0]<0])==.5
    assert residual==0
    return {'raw':[.25,.75,1.25],'feed':[0,1,1],'positive_absorbed_mass':.5,
            'upward_rounding_mass':.25,'downward_rounding_mass':.5,'endpoint_shadow_H':1.25,
            'endpoint_error_D':-.25,'endpoint_feed':1.0}

def read_state(run,name,source_rows):
    trace=json.loads((run/'trace.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    complete=json.loads((run/'complete.json').read_text())
    assert complete['status']=='complete' and complete['manifestSha256']==sha(run/'manifest.json')
    assert trace['selection']=={'layers':[0,7],'heads':[0]}
    spec=trace['arrays'][name]
    assert spec['dtype']=='float32' and spec['shape']==[783,1,2,1,32]
    file=run/spec['file'];identity=manifest['files'][spec['file']]
    assert file.stat().st_size==identity['bytes']==783*2*32*4 and sha(file)==identity['sha256']
    source_rows.append({'run':run.name,'array':name,'sha256':identity['sha256'],
                        'manifest_sha256':sha(run/'manifest.json')})
    return np.fromfile(file,dtype='<f4').reshape(783,2,32)

def first_crossing(curve,fraction):
    final=float(curve[-1])
    if final==0:return {'first_target_pixel_index':None,'later_below_target_count':None}
    oriented=np.asarray(curve)*math.copysign(1,final)
    target=abs(final)*fraction
    reached=np.flatnonzero(oriented>=target)
    if reached.size==0:return {'first_target_pixel_index':None,'later_below_target_count':None}
    first=int(reached[0]);return {'first_target_pixel_index':first+1,
                                  'later_below_target_count':int((oriented[first+1:]<target).sum())}

def main():
    p=argparse.ArgumentParser()
    for name in ('analysis-json','absolute-scores-csv','token-npz','runs-dir','range-json','output-dir'):
        p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();fixture=synthetic_checks()
    frozen=json.loads(args.analysis_json.read_text());source_range=json.loads(args.range_json.read_text())
    with np.load(args.token_npz,allow_pickle=False) as z:
        token=z['token_bits'].astype('float64');indices=z['indices'];arms=tuple(z['arms'])
    assert token.shape==(32,6,783) and arms==ARMS
    assert list(indices)==[r['index'] for r in frozen['rows']]
    with args.absolute_scores_csv.open() as f:
        absolute=list(csv.DictReader(f))
    cells={(int(r['index']),r['arm']):float(r['conditional_bits_per_predicted_pixel'])
           for r in absolute if r['cohort']=='fixed32'}
    assert len(cells)==192
    development_pair={r['arm']:Decimal(r['conditional_bits_per_predicted_pixel'])
                      for r in absolute if r['cohort']=='development10' and r['index']=='2'}
    pair_index2_delta=development_pair['bf16_pair_boundary']-development_pair['A00']
    assert pair_index2_delta==Decimal('6.60420120E-8')
    score=np.array([[cells[(row['index'],arm)] for arm in ARMS] for row in frozen['rows']],dtype='float64')
    assert all(score[i,j]==frozen['rows'][i]['scores'][arm] for i in range(32) for j,arm in enumerate(ARMS))
    max_token_endpoint_difference=float(np.max(np.abs(score-token.mean(axis=2))))
    assert max_token_endpoint_difference<1e-12
    score_results={'arms':{},'contrasts':{}}
    for j,arm in enumerate(ARMS):
        score_results['arms'][arm]={'score_descriptive':describe(score[:,j]),
                                    'signed_minus_FP32':describe(score[:,j]-score[:,0]) if arm=='A00' else stats(score[:,j]-score[:,0])}
    for name,pair in CONTRASTS.items():
        if pair is None:values=score[:,3]-score[:,1]-score[:,2]+score[:,0]
        else:values=score[:,pair[0]]-score[:,pair[1]]
        s=stats(values);original=frozen['summary']['contrasts'][FROZEN_NAMES[name]]
        assert abs(s['mean']-original['mean'])<1e-12
        s['original_pointwise_bootstrap_95']=original['bootstrap_95_pointwise']
        score_results['contrasts'][name]=s
    write_json(out/'score_statistics_v2.json',score_results)

    names=('A10_minus_A00','A01_minus_A00','A11_minus_A00','FP16_minus_A00','pair_minus_A00','primary_A11_minus_A10')
    pairs=((1,0),(2,0),(3,0),(4,0),(5,0),(3,1))
    curves=np.empty((32,6,783),dtype='float64')
    position_rows=[];window_rows=[];crossing_rows=[]
    for j,(name,(a,b)) in enumerate(zip(names,pairs)):
        step=token[:,a]-token[:,b]
        curves[:,j]=np.cumsum(step,axis=1)
        for t in range(783):
            values=curves[:,j,t]
            position_rows.append({'contrast':name,'target_pixel_index':t+1,
                                  'mean_cumulative_bits_per_image':float(values.mean()),
                                  'sample_sd_cumulative_bits_per_image':float(values.std(ddof=1)),
                                  'mean_cumulative_final_bpp_contribution':float(values.mean()/783)})
        for i,row in enumerate(frozen['rows']):
            c=curves[i,j];final=float(c[-1]);sign=math.copysign(1,final) if final else 0
            oriented=sign*c
            crossing={'index':row['index'],'label':row['label'],'contrast':name,
                      'final_signed_bits':final,'negative_step_changes':int((np.diff(oriented)<0).sum()),
                      'max_oriented_drawdown_bits':float(np.max(np.maximum.accumulate(oriented)-oriented))}
            for frac in FRACTIONS:
                got=first_crossing(c,frac)
                crossing[f'first_{int(frac*100)}pct_pixel']=got['first_target_pixel_index']
                crossing[f'later_below_{int(frac*100)}pct_steps']=got['later_below_target_count']
            crossing_rows.append(crossing)
            for lo,hi in WINDOWS:
                values=step[i,lo:hi]
                window_rows.append({'index':row['index'],'label':row['label'],'contrast':name,
                                    'step_start':lo,'step_end_exclusive':hi,'length':hi-lo,
                                    'signed_sum_bits':float(values.sum()),
                                    'signed_per_token_bits':float(values.mean()),
                                    'signed_final_bpp_contribution':float(values.sum()/783),
                                    'mean_absolute_per_token_bits':float(np.abs(values).mean())})
    write_csv(out/'position_curves_v2.csv',position_rows)
    write_csv(out/'position_windows_v2.csv',window_rows)
    write_csv(out/'position_crossings_v2.csv',crossing_rows)
    np.savez_compressed(out/'position_image_curves_v2.npz',indices=indices,contrasts=np.asarray(names),
                        cumulative_bits_per_image=curves)
    meanprimary=curves[:,5].mean(axis=0)
    mean_crossings={f'{int(frac*100)}pct':first_crossing(meanprimary,frac) for frac in FRACTIONS}
    mean_crossings['negative_step_changes']=int((np.diff(meanprimary)<0).sum())
    mean_crossings['max_drawdown_bits']=float(np.max(np.maximum.accumulate(meanprimary)-meanprimary))

    ratios_f=np.full((32,6,783,2,32),np.nan,dtype='float64')
    ratios_h=np.full_like(ratios_f,np.nan)
    image_mass_ratio_f=np.full((32,6,783),np.nan,dtype='float64')
    image_mass_ratio_h=np.full_like(image_mass_ratio_f,np.nan)
    coordinate_rows=[];mass_windows=[];source_rows=[];identical_layer0=[];max_residual=0
    for i,row in enumerate(frozen['rows']):
        baseline_layer0=None
        for j,arm in enumerate(ARMS):
            run=args.runs_dir/f"image_{row['index']:05d}_{arm}_v1"
            raw=read_state(run,'native_kc_state',source_rows)
            feed=read_state(run,'feed_kc_state',source_rows)
            if arm=='A01':baseline_layer0=(raw[:,0].copy(),feed[:,0].copy())
            if arm=='A11':
                assert baseline_layer0 is not None
                identical_layer0.append(bool(np.array_equal(raw[:,0],baseline_layer0[0]) and
                                             np.array_equal(feed[:,0],baseline_layer0[1])))
            d,e,H,D,absorbed,residual=mass(raw,feed)
            max_residual=max(max_residual,residual)
            f=feed.astype('float64')
            np.divide(D,f,out=ratios_f[i,j],where=f!=0)
            np.divide(D,H,out=ratios_h[i,j],where=H!=0)
            Dsum=D.sum(axis=(1,2));Fsum=f.sum(axis=(1,2));Hsum=H.sum(axis=(1,2))
            np.divide(Dsum,Fsum,out=image_mass_ratio_f[i,j],where=Fsum!=0)
            np.divide(Dsum,Hsum,out=image_mass_ratio_h[i,j],where=Hsum!=0)
            for layer_slot,layer in enumerate((0,7)):
                for coord in range(32):
                    ft=float(f[-1,layer_slot,coord]);ht=float(H[-1,layer_slot,coord]);dt=float(D[-1,layer_slot,coord])
                    coordinate_rows.append({'index':row['index'],'label':row['label'],'arm':arm,
                                            'layer':layer,'coordinate':coord,'final_feed':ft,'final_shadow_H':ht,
                                            'final_error_D':dt,'D_over_f':dt/ft if ft else None,
                                            'D_over_H':dt/ht if ht else None,
                                            'zero_f':int(ft==0),'zero_H':int(ht==0)})
                cumulative_absorbed=0.0;cumulative_up=0.0;cumulative_down=0.0
                for lo,hi in WINDOWS:
                    x=d[lo:hi,layer_slot];err=e[lo:hi,layer_slot];lost=absorbed[lo:hi,layer_slot]
                    absorbed_mass=float(x[lost].sum());up=float(err[err>0].sum());down=float((-err[err<0]).sum())
                    cumulative_absorbed+=absorbed_mass;cumulative_up+=up;cumulative_down+=down
                    Dend=float(D[hi-1,layer_slot].sum());Hend=float(H[hi-1,layer_slot].sum());Fend=float(f[hi-1,layer_slot].sum())
                    assert abs(Fend-Hend-Dend)<1e-7
                    mass_windows.append({'index':row['index'],'label':row['label'],'arm':arm,
                                         'layer':layer,'step_start':lo,'step_end_exclusive':hi,'length':hi-lo,
                                         'positive_raw_increment_count':int((x>0).sum()),
                                         'zero_raw_increment_count':int((x==0).sum()),
                                         'negative_raw_increment_count':int((x<0).sum()),
                                         'absorbed_positive_count':int(lost.sum()),
                                         'absorbed_positive_mass':absorbed_mass,
                                         'cumulative_absorbed_positive_mass':cumulative_absorbed,
                                         'positive_raw_increment_mass':float(x[x>0].sum()),
                                         'negative_raw_increment_mass':float((-x[x<0]).sum()),
                                         'rounding_up_mass':up,'rounding_down_mass':down,
                                         'cumulative_rounding_up_mass':cumulative_up,
                                         'cumulative_rounding_down_mass':cumulative_down,
                                         'signed_storage_error_mass':float(err.sum()),
                                         'endpoint_coordinate_sum_D':Dend,'endpoint_coordinate_sum_H':Hend,
                                         'endpoint_coordinate_sum_feed':Fend,
                                         'endpoint_ratio_of_sums_D_over_feed':Dend/Fend if Fend else None,
                                         'endpoint_ratio_of_sums_D_over_H':Dend/Hend if Hend else None})
    assert len(identical_layer0)==32
    write_csv(out/'state_coordinate_endpoints_v2.csv',coordinate_rows)
    write_csv(out/'state_mass_windows_v2.csv',mass_windows)
    write_csv(out/'state_source_files_v2.csv',source_rows)
    np.savez_compressed(out/'state_coordinate_ratios_v2.npz',indices=indices,arms=np.asarray(ARMS),
                        layer_numbers=np.asarray([0,7]),D_over_feed=ratios_f,D_over_shadow_H=ratios_h)
    np.savez_compressed(out/'state_image_mass_curves_v2.npz',indices=indices,arms=np.asarray(ARMS),
                        selected_layers=np.asarray([0,7]),ratio_of_selected_coordinate_sums_D_over_feed=image_mass_ratio_f,
                        ratio_of_selected_coordinate_sums_D_over_H=image_mass_ratio_h)
    mass_summary={}
    for j,arm in enumerate(ARMS):
        endpoints=[r for r in mass_windows if r['arm']==arm and r['step_end_exclusive']==783]
        assert len(endpoints)==64
        image_ratios_f=[];image_ratios_h=[]
        for row in frozen['rows']:
            selected=[r for r in endpoints if r['index']==row['index']]
            assert len(selected)==2
            Dsum=sum(r['endpoint_coordinate_sum_D'] for r in selected)
            Fsum=sum(r['endpoint_coordinate_sum_feed'] for r in selected)
            Hsum=sum(r['endpoint_coordinate_sum_H'] for r in selected)
            image_ratios_f.append(Dsum/Fsum)
            image_ratios_h.append(Dsum/Hsum)
        all_rows=[r for r in mass_windows if r['arm']==arm]
        pooled_D=sum(r['endpoint_coordinate_sum_D'] for r in endpoints)
        pooled_F=sum(r['endpoint_coordinate_sum_feed'] for r in endpoints)
        pooled_H=sum(r['endpoint_coordinate_sum_H'] for r in endpoints)
        mass_summary[arm]={'image_final_ratio_of_coordinate_sums_D_over_feed':describe(image_ratios_f),
                           'image_final_ratio_of_coordinate_sums_D_over_H':describe(image_ratios_h),
                           'pooled_final_D':pooled_D,'pooled_final_feed':pooled_F,'pooled_final_H':pooled_H,
                           'pooled_final_D_over_feed':pooled_D/pooled_F,'pooled_final_D_over_H':pooled_D/pooled_H,
                           'total_absorbed_positive_mass':sum(r['absorbed_positive_mass'] for r in all_rows),
                           'total_rounding_up_mass':sum(r['rounding_up_mass'] for r in all_rows),
                           'total_rounding_down_mass':sum(r['rounding_down_mass'] for r in all_rows),
                           'zero_feed_ratio_positions':int(np.isnan(ratios_f[:,j]).sum()),
                           'zero_shadow_ratio_positions':int(np.isnan(ratios_h[:,j]).sum())}
    range_rows=source_range['runs']
    original_max=max(range_rows,key=lambda r:r['full_state_range']['maxAbsReturned'])
    assert original_max['index']==3 and original_max['full_state_range']['maxAbsReturned']==14155.0224609375
    fixed32_fp16_ranges=[]
    for row in frozen['rows']:
        run=args.runs_dir/f"image_{row['index']:05d}_fp16_boundary_v1"
        trace=json.loads((run/'trace.json').read_text())
        fixed32_fp16_ranges.append({'index':row['index'],'maxAbsReturned':trace['range']['maxAbsReturned'],
                                    'inspectedCalls':trace['range']['inspectedCalls']})
    write_csv(out/'fixed32_fp16_runner_ranges_v2.csv',fixed32_fp16_ranges)
    result={'status':'retrospective_saved_data','n_images':32,'n_arms':6,'state_scope':'selected layers 0 and 7, head 0, normalizer kc only',
            'score_source':'literal fixed32 conditional_bits_per_predicted_pixel cells in absolute_scores.csv; token means are endpoint QA only',
            'max_absolute_token_endpoint_score_difference':max_token_endpoint_difference,
            'development_index2_pair_minus_FP32_from_literal_CSV':str(pair_index2_delta),
            'state_identity':'raw graph return r; encoded feed f; previous feed at token0 exactly zero; d=r-fprev; e=f-r; H=cumsum(d); D=cumsum(e); f=H+D',
            'synthetic_fixture':fixture,'max_absolute_identity_residual':max_residual,
            'identical_A01_A11_layer0_raw_and_feed_images':sum(identical_layer0),
            'selected_normalizer_mass_summary':mass_summary,
            'position_primary_mean_curve_crossings':mean_crossings,
            'original_14155_provenance':{'max':14155.0224609375,'development_image_index':3,'runs':4,'calls_per_run':783,
                                        'states':'full returned kc and kv before FP16 encoding; native runner range log',
                                        'not_a_future_overflow_bound':True},
            'fixed32_fp16_max_recorded':max(fixed32_fp16_ranges,key=lambda r:r['maxAbsReturned']),
            'fixed32_full_state_range_log_scope':'FP16 arm only; other fixed32 arm trace files lack full-state range logs',
            'source_sha256':{args.analysis_json.name:sha(args.analysis_json),args.absolute_scores_csv.name:sha(args.absolute_scores_csv),
                             args.token_npz.name:sha(args.token_npz),args.range_json.name:sha(args.range_json)},
            'elapsed_seconds':time.monotonic()-started}
    outputs=('score_statistics_v2.json','position_curves_v2.csv','position_windows_v2.csv',
             'position_crossings_v2.csv','position_image_curves_v2.npz',
             'state_coordinate_endpoints_v2.csv','state_mass_windows_v2.csv','state_source_files_v2.csv',
             'state_coordinate_ratios_v2.npz','state_image_mass_curves_v2.npz','fixed32_fp16_runner_ranges_v2.csv')
    result['outputs_sha256']={name:sha(out/name) for name in outputs}
    write_json(out/'summary_v2.json',result)
    print(json.dumps({'status':'ok','seconds':result['elapsed_seconds'],'max_identity_residual':max_residual,
                      'layer0_identical':sum(identical_layer0),'fixed32_fp16_max':result['fixed32_fp16_max_recorded']}))

if __name__=='__main__':main()
