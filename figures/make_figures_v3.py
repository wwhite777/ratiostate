#!/usr/bin/env python3
"""Reproducible figures from frozen fixed32 scores and retrospective saved-state analysis."""
import argparse
import csv
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS=('A00','A10','A01','A11','fp16_boundary','bf16_pair_boundary')
COL={'A00':'#666666','A10':'#0072B2','A01':'#009E73','A11':'#CC79A7',
     'fp16_boundary':'#D55E00','bf16_pair_boundary':'#56B4E9'}
NAME={'A00':'FP32','A10':'S-BF16','A01':'z-BF16','A11':'Joint BF16',
      'fp16_boundary':'FP16','bf16_pair_boundary':'Two-word BF16'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,
                     'axes.titlesize':12,'xtick.labelsize':11,'ytick.labelsize':11,
                     'legend.fontsize':11,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jitter(index,key):
    h=hashlib.sha256(f'{index}|{key}|RatioState-submission-v3'.encode()).digest()
    return (int.from_bytes(h[:8],'big')/2**64-.5)*.27
def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=300,facecolor='white')
    fig.savefig(out/f'{name}.pdf',facecolor='white',metadata={
        'CreationDate':datetime(2026,9,26,tzinfo=timezone.utc),
        'ModDate':datetime(2026,9,26,tzinfo=timezone.utc)})
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();data=args.data_dir.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    required=('frozen_analysis_v3.json','absolute_scores.csv','score_statistics_v2.json',
              'state_mass_windows_v2.csv','state_image_mass_curves_v2.npz',
              'position_image_curves_v2.npz','summary_v2.json')
    assert all((data/n).is_file() for n in required)
    frozen=json.loads((data/'frozen_analysis_v3.json').read_text())
    stats=json.loads((data/'score_statistics_v2.json').read_text())
    with (data/'absolute_scores.csv').open() as f:raw=list(csv.DictReader(f))
    cells={(int(r['index']),r['arm']):float(r['conditional_bits_per_predicted_pixel']) for r in raw if r['cohort']=='fixed32'}
    assert len(cells)==192
    rows=frozen['rows'];assert len(rows)==32
    indices=[r['index'] for r in rows]

    # F1: original four contrast definitions, pointwise resampling bounds, and all paired images.
    definitions=[('normalizer_given_bf16_matrix','Primary: Joint BF16\n− S-BF16','#0072B2'),
                 ('normalizer_only','Secondary: z-BF16\n− FP32','#555555'),
                 ('interaction','Secondary: interaction','#555555'),
                 ('A10_minus_fp16','Secondary: S-BF16\n− FP16','#555555')]
    fig,ax=plt.subplots(figsize=(7.3,5.0));fig.subplots_adjust(left=.39,right=.95,bottom=.17,top=.80)
    plotted=[]
    for k,(key,label,color) in enumerate(definitions):
        y=3-k;vals=np.array([r['contrasts'][key] for r in rows])
        s=frozen['summary']['contrasts'][key];lo,hi=s['bootstrap_95_pointwise']
        assert abs(vals.mean()-s['mean'])<1e-14
        for row,value in zip(rows,vals):
            yy=y+jitter(row['index'],key)
            ax.scatter(value,yy,s=21,facecolor=color,alpha=.34,edgecolor='none',zorder=2)
            plotted.append({'contrast':key,'index':row['index'],'label':row['label'],
                            'signed_change_bpp':float(value),'plot_y':yy,'mean_bpp':s['mean'],
                            'original_resampling_lower':lo,'original_resampling_upper':hi})
        ax.plot([lo,hi],[y,y],color=color,lw=2.5,zorder=4)
        for x in (lo,hi):ax.plot([x,x],[y-.11,y+.11],color=color,lw=2,zorder=4)
        ax.scatter(s['mean'],y,s=82,marker='D',color=color,edgecolor='white',lw=.8,zorder=5)
    ax.axvline(0,color='#222222',lw=1.1)
    ax.set(yticks=[3,2,1,0],yticklabels=[v[1] for v in definitions],xlim=(-.0195,.025),ylim=(-.55,3.55),
           xlabel='Signed conditional score change (bits per predicted pixel)')
    ax.set_xticks([-.01,0,.01,.02]);ax.grid(axis='x',color='#dddddd',lw=.7);ax.set_axisbelow(True)
    fig.text(.5,.975,'Paired score contrasts\nDots: images; diamonds: means; bars: approximate 95% CIs',
             ha='center',va='top',fontsize=11)
    write_csv(out/'primary_contrasts_v3_plotted.csv',plotted);save(fig,out,'primary_contrasts_v3')

    # F2: signed paired points and means ± SE; SD and MAE are separate descriptive quantities.
    fmt=ARMS[1:];disp=['S-BF16','z-BF16','Joint BF16','FP16','Two-word\nBF16']
    fig,(ax,lower)=plt.subplots(2,1,figsize=(7.3,6.4),sharex=True,
                                gridspec_kw={'height_ratios':[2.35,1.25]},layout='constrained')
    plotted=[]
    for k,arm in enumerate(fmt):
        values=np.array([cells[(idx,arm)]-cells[(idx,'A00')] for idx in indices])
        s=stats['arms'][arm]['signed_minus_FP32']
        assert abs(values.mean()-s['mean'])<1e-14
        for idx,value in zip(indices,values):
            xx=k+jitter(idx,arm)
            ax.scatter(xx,value,s=22,color=COL[arm],alpha=.53,edgecolor='none',zorder=2)
            plotted.append({'index':idx,'arm':arm,'signed_minus_FP32_bpp':float(value),
                            'absolute_minus_FP32_bpp':float(abs(value)),'plot_x':xx,
                            'mean':s['mean'],'sample_sd':s['sample_sd'],
                            'standard_error':s['standard_error'],'mean_absolute':s['mean_absolute']})
        ax.errorbar(k,s['mean'],yerr=s['standard_error'],fmt='D',markersize=8,color='#111111',
                    ecolor='#111111',elinewidth=2,capsize=4,markeredgecolor='white',zorder=5)
        lower.scatter(k-.10,s['mean_absolute'],s=77,marker='o',color=COL[arm],edgecolor='#222222',lw=.5,zorder=4)
        lower.scatter(k+.10,s['sample_sd'],s=72,marker='s',facecolor='white',edgecolor=COL[arm],lw=1.8,zorder=4)
    ax.axhline(0,color='#222222',lw=1.1)
    ax.set(ylabel='Signed change from FP32\n(bits per predicted pixel)',ylim=(-.019,.036))
    ax.grid(axis='y',color='#dddddd',lw=.7)
    ax.text(.02,.97,'32 paired images  •  diamonds: mean ± SE',transform=ax.transAxes,ha='left',va='top',fontsize=11,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.88})
    lower.set_yscale('log');lower.set(ylim=(1.4e-6,.035),xticks=range(5),xticklabels=disp,
                                  ylabel='Dispersion and absolute change\n(log scale)')
    lower.grid(axis='y',which='major',color='#dddddd',lw=.7);lower.set_axisbelow(True)
    lower.scatter([],[],marker='o',color='#777777',label='mean |Δ|')
    lower.scatter([],[],marker='s',facecolor='white',edgecolor='#777777',label='sample SD')
    lower.legend(loc='lower left',bbox_to_anchor=(.02,.06),frameon=True,
                 facecolor='white',edgecolor='#cccccc',ncol=2)
    write_csv(out/'score_dispersion_v3_plotted.csv',plotted);save(fig,out,'score_dispersion_v3')

    # F3: event rates, unweighted selected-normalizer mass, and score trajectories.
    with (data/'state_mass_windows_v2.csv').open() as f:mass=list(csv.DictReader(f))
    with np.load(data/'state_image_mass_curves_v2.npz',allow_pickle=False) as z:
        mass_ratio=z['ratio_of_selected_coordinate_sums_D_over_feed'];mass_indices=list(z['indices']);mass_arms=list(z['arms'])
    with np.load(data/'position_image_curves_v2.npz',allow_pickle=False) as z:
        score_curves=z['cumulative_bits_per_image'];score_indices=list(z['indices']);score_names=list(z['contrasts'])
    assert mass_ratio.shape==(32,6,783) and mass_indices==score_indices==indices and mass_arms==list(ARMS)
    assert score_curves.shape==(32,6,783)
    plt.rcParams.update({'font.size':12.5,'axes.labelsize':12.5,'axes.titlesize':13,
                         'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12})
    fig=plt.figure(figsize=(7.3,8.5))
    gs=fig.add_gridspec(3,2,left=.16,right=.97,bottom=.095,top=.87,
                        hspace=.50,wspace=.30,height_ratios=[1,1.05,1.12])
    top=(fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]))
    mid=fig.add_subplot(gs[1,:]);bot=fig.add_subplot(gs[2,:])
    groups=[('FP32 / S-BF16',('A00','A10'),'#666666','-','o'),
            ('Joint BF16',('A11',),'#CC79A7','-','s'),
            ('z-BF16',('A01',),'#009E73','--','o'),
            ('FP16',('fp16_boundary',),'#D55E00','-','^'),
            ('Two-word BF16',('bf16_pair_boundary',),'#0072B2','-','D')]
    event_rows=[]
    for layer,ax in zip((0,7),top):
        for label,members,color,style,marker in groups:
            means=[];mids=[]
            for lo,hi in ((0,128),(128,256),(256,512),(512,783)):
                selected=[r for r in mass if r['arm']==members[0] and int(r['layer'])==layer and int(r['step_start'])==lo]
                assert len(selected)==32
                rates=np.array([int(r['absorbed_positive_count'])/int(r['positive_raw_increment_count']) for r in selected])
                mean=float(rates.mean());count=sum(int(r['absorbed_positive_count']) for r in selected)
                if len(members)==2:assert mean==0
                means.append(mean);mids.append((lo+hi-1)/2)
                for member in members:
                    event_rows.append({'arm':member,'layer':layer,'step_start':lo,'step_end_exclusive':hi,
                                       'mean_image_absorbed_positive_rate':mean,'pooled_absorbed_positive_count':count})
            ax.plot(mids,means,color=color,ls=style,marker=marker,lw=1.8,markersize=5.5,
                    markerfacecolor='white' if members==('A01',) else color,
                    markeredgewidth=1.5 if members==('A01',) else .6,label=label)
        ax.set(title=f'Layer {layer}, head 0',ylim=(-.04,.94),xlim=(0,783),xticks=[64,192,384,647],
               xticklabels=['0–127','128–255','256–511','512–782'])
        ax.grid(axis='y',color='#dddddd',lw=.7);ax.tick_params(axis='x',rotation=30)
    top[0].set_ylabel('Mean image fraction of\npositive increments absorbed')
    handles,labels=top[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,
               columnspacing=.8,handlelength=1.8)
    mass_rows=[]
    for j,arm in enumerate(ARMS):
        values=mass_ratio[:,j]
        mean=np.nanmean(values,axis=0)
        if arm not in ('A10',):
            mid.plot(np.arange(1,784),mean,color=COL[arm],lw=2.0,
                     ls='--' if arm=='A01' else '-',label=NAME[arm])
        for t,value in enumerate(mean,1):
            mass_rows.append({'arm':arm,'completed_call_index':t,
                              'mean_image_ratio_of_selected_coordinate_sums_D_over_feed':float(value)})
    mid.axhline(0,color='#222222',lw=1.0)
    mid.set(xlim=(1,783),ylabel='Mean selected z mass ratio\nD / sum(feed)')
    mid.grid(axis='y',color='#dddddd',lw=.7)
    score_labels={'A10_minus_A00':'S-BF16 − FP32','A01_minus_A00':'z-BF16 − FP32',
                  'A11_minus_A00':'Joint BF16 − FP32','FP16_minus_A00':'FP16 − FP32',
                  'pair_minus_A00':'Two-word − FP32','primary_A11_minus_A10':'Joint BF16 − S-BF16'}
    score_colors={'A10_minus_A00':COL['A10'],'A01_minus_A00':COL['A01'],'A11_minus_A00':COL['A11'],
                  'FP16_minus_A00':COL['fp16_boundary'],'pair_minus_A00':COL['bf16_pair_boundary'],
                  'primary_A11_minus_A10':'#111111'}
    score_rows=[]
    for j,name in enumerate(score_names):
        mean=score_curves[:,j].mean(axis=0)
        bot.plot(np.arange(1,784),mean,color=score_colors[name],lw=2.0,
                 ls='--' if name=='primary_A11_minus_A10' else '-',label=score_labels[name])
        for t,value in enumerate(mean,1):
            score_rows.append({'contrast':name,'predicted_pixel_index':t,'mean_cumulative_bits_per_image':float(value)})
    bot.axhline(0,color='#222222',lw=1.0)
    bot.set(xlim=(1,783),xlabel='Completed graph call / predicted pixel index (t + 1)',
            ylabel='Mean cumulative score change\n(bits per image)')
    bot.grid(axis='y',color='#dddddd',lw=.7)
    score_handles,score_legend_labels=bot.get_legend_handles_labels()
    fig.legend(score_handles,score_legend_labels,loc='center',bbox_to_anchor=(.57,.33),
               ncol=3,frameon=False,fontsize=11.5,columnspacing=.75,handlelength=1.8)
    write_csv(out/'position_mass_score_v3_events_plotted.csv',event_rows)
    write_csv(out/'position_mass_score_v3_mass_plotted.csv',mass_rows)
    write_csv(out/'position_mass_score_v3_score_plotted.csv',score_rows)
    save(fig,out,'position_mass_score_v3')

    outputs=('primary_contrasts_v3.png','primary_contrasts_v3.pdf','primary_contrasts_v3_plotted.csv',
             'score_dispersion_v3.png','score_dispersion_v3.pdf','score_dispersion_v3_plotted.csv',
             'position_mass_score_v3.png','position_mass_score_v3.pdf',
             'position_mass_score_v3_events_plotted.csv','position_mass_score_v3_mass_plotted.csv',
             'position_mass_score_v3_score_plotted.csv')
    receipt={'status':'rendered','script_sha256':sha(Path(__file__)),
             'sources_sha256':{n:sha(data/n) for n in required},
             'output_sha256':{n:sha(out/n) for n in outputs},
             'matplotlib_version':matplotlib.__version__,'width_inches':7.3,'nominal_min_label_font_points':11}
    (out/'figure_receipt_v3.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'rendered','files':len(outputs)}))

if __name__=='__main__':main()
