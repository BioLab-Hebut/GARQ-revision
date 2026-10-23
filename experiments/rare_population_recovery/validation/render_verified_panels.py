"""Render four prespecified vector panels only from complete validated aggregates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

COLORS=['#247F86','#D87945','#737F8C']
MARKERS=['o','s','^']
METHODS=['GARQ_quantizer','SEACells','KMeans']
LABELS=['GARQ quantizer','SEACells','KMeans']
FRACTIONS=[1.,.5,.25,.1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--analysis-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    metadata=json.loads((args.analysis_dir/'analysis_metadata.json').read_text(encoding='utf-8'))
    if metadata['status']!='COMPLETE_VALIDATED' or metadata['fits']!=144 or metadata['method_conditions']!=48:
        raise ValueError('The full independently verified 144-run analysis is required')
    path=args.analysis_dir/'condition_summary_48.csv'
    if sha(path)!=metadata['output_sha256'][path.name]:raise ValueError('Aggregate table was altered after verification')
    with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=48:raise ValueError('Expected all 48 method conditions')
    lookup={(r['dataset'],int(r['compression_setting']),r['method'],float(r['retained_fraction'])):r for r in rows}
    if len(lookup)!=48:raise ValueError('Duplicate condition rows')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':13,'axes.titlesize':13,
                         'xtick.labelsize':11.5,'ytick.labelsize':11.5,'pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white','axes.edgecolor':'#555555'})
    lower=min(float(r['rare_macro_F1_mean'])-float(r['rare_macro_F1_sd']) for r in rows)
    upper=max(float(r['rare_macro_F1_mean'])+float(r['rare_macro_F1_sd']) for r in rows)
    y_limits=(min(-.03,lower-.03),max(1.03,upper+.03))
    fig,axes=plt.subplots(2,2,figsize=(8.4,7.8))
    fig.subplots_adjust(left=.105,right=.975,bottom=.13,top=.87,wspace=.37,hspace=.66)
    fig.legend([Line2D([0],[0],color=c,marker=m,lw=1.5,ms=6) for c,m in zip(COLORS,MARKERS)],LABELS,
               ncol=3,loc='upper center',bbox_to_anchor=(.54,1.005),frameon=False,fontsize=12)
    for ax,(letter,dataset,compression) in zip(axes.flat,[('e','D5',30),('f','D5',50),('g','D11',30),('h','D11',50)]):
        for method,color,marker in zip(METHODS,COLORS,MARKERS):
            selected=[lookup[(dataset,compression,method,fraction)] for fraction in FRACTIONS]
            if any(int(r['n_runs'])!=3 for r in selected):raise ValueError('Exactly three runs per point required')
            values=[float(r['rare_macro_F1_mean']) for r in selected]
            errors=[float(r['rare_macro_F1_sd']) for r in selected]
            ax.errorbar(range(4),values,yerr=errors,color=color,marker=marker,lw=1.6,ms=5.5,
                        elinewidth=1.0,capsize=2.5,capthick=1.0,zorder=3)
        ax.set_xticks(range(4),['100','50','25','10'])
        ax.set_xlabel('Rare-population retention (%)')
        ax.set_ylabel('Rare-population macro-F1')
        ax.set_title(f'{dataset} | Compression ratio {compression}',pad=10)
        ax.text(-.21,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=14,va='bottom')
        ax.set_xlim(-.18,3.18);ax.set_ylim(y_limits);ax.set_yticks([0,.25,.5,.75,1])
        ax.tick_params(direction='out',length=3,width=.8);ax.grid(False)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    prefix=args.output_dir/'Figure2_efgh_preconstruction_rare_recovery'
    for suffix in ['pdf','svg','png']:
        fig.savefig(prefix.with_suffix('.'+suffix),dpi=180 if suffix=='png' else None)
    plt.close(fig)
    record={'status':'RENDERED_REQUIRES_VISUAL_REVIEW','plan_sha256':metadata['plan_sha256'],
            'analysis_metadata_sha256':sha(args.analysis_dir/'analysis_metadata.json'),
            'primary_metric':'rare_macro_F1','points':'mean of three paired sampling/aggregation runs',
            'error_bars':'sample SD across the same three runs','same_y_limits':list(y_limits),
            'color_and_shape_encoding':dict(zip(LABELS,[dict(color=c,marker=m) for c,m in zip(COLORS,MARKERS)])),
            'rasterized_artists':False,'statistical_significance_claims':False,'scientific_source_edited':False,
            'output_sha256':{prefix.with_suffix('.'+s).name:sha(prefix.with_suffix('.'+s)) for s in ['pdf','svg','png']},
            'script_sha256':sha(__file__)}
    (args.output_dir/'FIGURE_PROVENANCE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'output_prefix':str(prefix)},indent=2))

if __name__=='__main__':main()
