"""Read the user-specified UAD evidence tree; keep test/training provenance distinct."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('D:/CODE/Doing/WHU_Code/WHU')
OUT=ROOT/'output/training_log_review_20261105'
qg=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
def write_csv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)

def parse(path):
    text=path.read_text(encoding='utf-8-sig')
    lines=text.splitlines()
    training=[];evaluations=[];timings=[];pairs={};ag={};evaluation=None;row=None
    for i,line in enumerate(lines,1):
        match=re.search(r'Epoch\[(\d+)\] Iteration\[(\d+)/(\d+)\] Loss: ([\d.e+-]+), Acc: ([\d.e+-]+), Base Lr: ([\d.e+-]+)',line)
        if match:
            e,step,_,loss,acc,lr=match.groups()
            row=dict(epoch=int(e),iteration=int(step),loss=float(loss),accuracy=float(acc),lr=float(lr))
            training.append(row)
        match=re.search(r'Epoch\[(\d+)\] ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)$',line)
        if match:
            row.update(zip(('ce','wrt','pca','gpd'),map(float,match.groups()[1:])))
        match=re.search(r'Pair \(Q:(\d), G:(\d)\) -> Rank-1: ([\d.]+)%, mAP: ([\d.]+)%, mINP: ([\d.]+)%',line)
        if match:
            query,gallery,r1,ap,inp=match.groups()
            pairs[query+'_'+gallery]=dict(rank1=float(r1),map=float(ap),minp=float(inp),line=i)
        match=re.search(r'(A->G|G->A) -> mAP: ([\d.]+)%, Rank-1: ([\d.]+)%, Rank-10: ([\d.]+)%',line)
        if match:
            direction,ap,r1,r10=match.groups();ag[direction]=dict(map=float(ap),rank1=float(r1),rank10=float(r10))
        match=re.search(r'Validation Results(?: - Epoch: (\d+))?',line)
        if match:
            evaluation=dict(epoch=int(match[1]) if match[1] else None,pairs=pairs,ag=ag,line=i)
            evaluations.append(evaluation);pairs={};ag={}
        match=re.search(r'INFO: mAP: ([\d.]+)%',line)
        if match and evaluation is not None:
            evaluation['map']=float(match[1])
        match=re.search(r'CMC curve, Rank-(\d+)\s*:\s*([\d.]+)%',line)
        if match and evaluation is not None:
            evaluation['rank'+match[1]]=float(match[2])
        match=re.search(r'Epoch (\d+) done. Time per batch: ([\d.]+)\[s\] Total: ([\d.]+)\[s\]',line)
        if match:
            timings.append(dict(epoch=int(match[1]),per_batch=float(match[2]),seconds=float(match[3])))
    return dict(path=str(path),lines=len(lines),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                training=training,evaluations=evaluations,timings=timings)

baseline_files={p.relative_to(BASE/'logs').as_posix():parse(p) for p in sorted((BASE/'logs').rglob('*.txt'))}
train=baseline_files['1000/1000_train_log_20260911_2145.txt']
test=baseline_files['1000/1000_test_log_20260912_1037.txt']
final=test['evaluations'][-1]
assert all(train['evaluations'][-1][key]==final[key] for key in ('map','rank1','rank5','rank10','ag'))
assert all(all(train['evaluations'][-1]['pairs'][d][m]==final['pairs'][d][m] for m in ('rank1','map','minp')) for d in final['pairs'])

def flatten_config(path):
    text=path.read_text(encoding='utf-8-sig')
    body=text.split('Running with config:\n',1)[1]
    lines=[]
    for line in body.splitlines():
        if re.match(r'^\d{4}-\d\d-\d\d ',line):break
        lines.append(line)
    values=yaml.safe_load('\n'.join(lines))
    result={}
    def flatten(node,prefix=''):
        for key,value in node.items():
            if isinstance(value,dict):flatten(value,prefix+key+'.')
            else:result[prefix+key]=value
    flatten(values)
    return result
left=flatten_config(BASE/'logs/1000/1000_train_log_20260911_2145.txt')
right=flatten_config(ROOT/'logs/train_log_20261105_0937.txt')
common_config_changes={key:dict(baseline=left[key],qg=right[key]) for key in left if key in right and left[key]!=right[key]}

pair_comparison=[]
paper_lookup={r['direction']:r for r in qg['paper_comparison']}
for direction,base in final['pairs'].items():
    row=dict(direction=direction)
    for metric in ('rank1','map','minp'):
        row['paper_'+metric]=paper_lookup[direction]['paper_'+metric]
        row['baseline_'+metric]=base[metric]
        row['qg_'+metric]=qg['final_evaluation']['pairs'][direction][metric]
        row['delta_'+metric]=round(row['qg_'+metric]-base[metric],2)
    pair_comparison.append(row)
overall=[]
for metric,ref in [('map',11.),('rank1',29.5),('rank5',46.3),('rank10',54.8)]:
    overall.append(dict(metric=metric,paper=ref,baseline=final[metric],qg=qg['final_evaluation'][metric],
                        delta=round(qg['final_evaluation'][metric]-final[metric],2)))
ag=[]
for direction,base in final['ag'].items():
    ref={'A->G':dict(map=11.5,rank1=31.9,rank10=58.3),'G->A':dict(map=13.3,rank1=30.5,rank10=54.7)}[direction]
    for metric in ('map','rank1','rank10'):
        ag.append(dict(direction=direction,metric=metric,paper=ref[metric],baseline=base[metric],
                       qg=qg['final_evaluation']['ag'][direction][metric],
                       delta=round(qg['final_evaluation']['ag'][direction][metric]-base[metric],2)))
table3=[]
for name,file,paper in [
    ('WHU-MARS-1000','1000/1000_test_log_20260912_1037.txt',(11.,29.5,46.3,54.8)),
    ('WHU-MARS-1000-GD','1000GD/1000GD_test_log_20260912_1045.txt',(14.5,31.9,45.,52.5)),
    ('WHU-MARS-2337','2337/2337_test_log_20261025_1505.txt',(9.6,25.7,40.9,48.2))]:
    base=baseline_files[file]['evaluations'][-1]
    for metric,reference in zip(('map','rank1','rank5','rank10'),paper):
        table3.append(dict(dataset=name,metric=metric,paper=reference,baseline=base[metric],
                           qg=qg['final_evaluation'][metric] if name=='WHU-MARS-1000' else None))

write_csv('baseline_training.csv',train['training'])
write_csv('baseline_evaluation_overall.csv',[{k:v for k,v in r.items() if k not in ('pairs','ag')} for r in train['evaluations']])
write_csv('matched_baseline_pairs.csv',pair_comparison)
write_csv('matched_baseline_overall.csv',overall)
write_csv('matched_baseline_ag.csv',ag)
write_csv('table3_three_way.csv',table3)
source_files=['datasets/whu_mars.py','datasets/sampler.py','datasets/sampler_ddp.py','loss/make_loss.py',
              'loss/wrt_loss.py','solver/scheduler_factory.py','utils/metrics.py']
report=dict(
    files=[{k:v for k,v in data.items() if k not in ('training','evaluations','timings')} for data in baseline_files.values()],
    common_config_changes=common_config_changes,
    original_source_identical={n:(BASE/n).read_bytes()==(ROOT/'WHU'/n).read_bytes() for n in source_files},
    baseline_train_and_standalone_test_agree=True,overall=overall,pairs=pair_comparison,ag=ag,table3=table3,
    baseline_final_training=train['training'][-1],
    baseline_train_seconds=sum(r['seconds'] for r in train['timings']),
    baseline_median_seconds_per_batch=statistics.median(r['per_batch'] for r in train['timings']),
    qg_train_seconds=qg['training_seconds'],
    baseline_1000GD=baseline_files['1000GD/1000GD_test_log_20260912_1045.txt']['evaluations'][-1],
    baseline_2337=baseline_files['2337/2337_test_log_20261025_1505.txt']['evaluations'][-1],
    notes=['2337train_1000test is not used as the WHU-MARS-1000-trained baseline.',
           'Historical baseline has no explicit AMP skip log; original code updates GPD bank even if SGD is skipped.',
           'Matched main training configuration; test batch 512 vs 1024 and AMP/checkpoint/resume code differ.'])
(OUT/'baseline_comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,metric,title in [(axes[0,0],'rank1','Overall Rank-1'),(axes[0,1],'map','Overall mAP')]:
    axes_rows=train['evaluations']
    ax.plot([r['epoch'] for r in axes_rows],[r[metric] for r in axes_rows],marker='o',ms=3,label='UAD reproduction')
    rows=list(csv.DictReader((OUT/'evaluation_overall.csv').open(encoding='utf-8-sig')))
    ax.plot([int(r['epoch']) for r in rows],[float(r[metric]) for r in rows],marker='o',ms=3,label='UAD + QG-SFM')
    ax.set(title=title,xlabel='Epoch',ylabel='Score (%)');ax.legend()
for rows,label in [(train['training'],'UAD reproduction'),(list(csv.DictReader((OUT/'training.csv').open(encoding='utf-8-sig'))),'UAD + QG-SFM')]:
    last={int(r['epoch']):r for r in rows}
    axes[1,0].plot(list(last),[float(r['loss']) for r in last.values()],label=label)
    axes[1,1].plot(list(last),[float(r['accuracy'])*100 for r in last.values()],label=label)
axes[1,0].set(title='Total training loss (iteration 100)',xlabel='Epoch',ylabel='Loss',yscale='log')
axes[1,1].set(title='Training accuracy (iteration 100)',xlabel='Epoch',ylabel='Accuracy (%)')
axes[1,0].legend();axes[1,1].legend()
fig.savefig(OUT/'baseline_training_comparison.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,metric,title in zip(axes,('rank1','map','minp'),('Rank-1','mAP','mINP')):
    matrix=np.array([r['delta_'+metric] for r in pair_comparison]).reshape(3,3)
    bound=max(abs(matrix).max(),.1)
    im=ax.imshow(matrix,cmap='RdBu',vmin=-bound,vmax=bound)
    ax.set(title=title+' change vs UAD (pp)',xticks=range(3),yticks=range(3),
           xticklabels=['RGB','NIR','TIR'],yticklabels=['RGB','NIR','TIR'],xlabel='Gallery',ylabel='Query')
    for i in range(3):
        for j in range(3):
            ax.text(j,i,'{:+.2f}'.format(matrix[i,j]),ha='center',va='center',
                    color='white' if abs(matrix[i,j])>.65*bound else 'black',fontsize=12)
    fig.colorbar(im,ax=ax,shrink=.7)
fig.suptitle('QG-SFM vs actual UAD reproduction: small, mixed changes',fontsize=12)
fig.savefig(OUT/'matched_baseline_delta.png',dpi=170);plt.close(fig)
print(json.dumps({k:report[k] for k in ('common_config_changes','overall','pairs','ag','baseline_final_training',
                                       'baseline_train_seconds','baseline_median_seconds_per_batch')},indent=2))
