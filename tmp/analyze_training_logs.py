"""Read every supplied training-log line; export research analysis, no source edits."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/training_log_review_20261105'
OUT.mkdir(parents=True, exist_ok=True)

def write_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

training, qg, evaluations, checkpoints, overflows, resumes, timings = [], [], [], [], [], [], []
files = []
for path in sorted((ROOT/'logs').glob('train_log_*.txt')):
    text = path.read_text(encoding='utf-8-sig')
    lines = text.splitlines()
    row, pending_pairs, pending_ag, evaluation = None, {}, {}, None
    for line_number, line in enumerate(lines, 1):
        meta = dict(source=path.name, line=line_number, time=line[:23])
        match = re.search(r'Epoch\[(\d+)\] Iteration\[(\d+)/(\d+)\] Loss: ([\d.e+-]+), Acc: ([\d.e+-]+), Base Lr: ([\d.e+-]+)', line)
        if match:
            e, i, n, loss, acc, lr = match.groups()
            row = dict(meta, epoch=int(e), iteration=int(i), estimated_batches=int(n),
                       loss=float(loss), accuracy=float(acc), lr=float(lr))
            training.append(row)
            continue
        match = re.search(r'Epoch\[(\d+)\] ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)$', line)
        if match:
            assert row and row['epoch'] == int(match[1])
            row.update(zip(('ce','wrt','pca','gpd'), map(float, match.groups()[1:])))
            row['reconstructed_loss'] = row['ce'] + row['wrt'] + .01*row['pca'] + row['gpd']
            row['loss_rounding_residual'] = row['loss'] - row['reconstructed_loss']
            continue
        match = re.search(r'QG layer (\d+) rank0-local: raw=\(([-\d.e+]+), ([-\d.e+]+)\), effective=\(([-\d.e+]+), ([-\d.e+]+)\), R=([-\d.e+]+)', line)
        if match:
            layer, a, b, alpha, beta, ratio = match.groups()
            qg.append(dict(meta, epoch=row['epoch'], iteration=row['iteration'], layer=int(layer),
                           raw_alpha=float(a), raw_beta=float(b), alpha=float(alpha), beta=float(beta), ratio=float(ratio),
                           alpha_derivative=1-(float(alpha)/.2)**2, beta_derivative=1-(float(beta)/.2)**2))
            continue
        match = re.search(r'Pair \(Q:(\d), G:(\d)\) -> Rank-1: ([\d.]+)%, mAP: ([\d.]+)%, mINP: ([\d.]+)%', line)
        if match:
            query, gallery, r1, ap, inp = match.groups()
            pending_pairs[query+'_'+gallery] = dict(rank1=float(r1), map=float(ap), minp=float(inp), line=line_number)
            continue
        match = re.search(r'(A->G|G->A) -> mAP: ([\d.]+)%, Rank-1: ([\d.]+)%, Rank-10: ([\d.]+)%', line)
        if match:
            direction, ap, r1, r10 = match.groups()
            pending_ag[direction] = dict(map=float(ap), rank1=float(r1), rank10=float(r10))
            continue
        match = re.search(r'Validation Results - Epoch: (\d+)', line)
        if match:
            evaluation = dict(meta, epoch=int(match[1]), pairs=pending_pairs, ag=pending_ag)
            evaluations.append(evaluation)
            pending_pairs, pending_ag = {}, {}
            continue
        match = re.search(r'INFO: mAP: ([\d.]+)%', line)
        if match and evaluation is not None:
            evaluation['map'] = float(match[1])
        match = re.search(r'CMC curve, Rank-(\d+)\s*: ([\d.]+)%', line)
        if match and evaluation is not None:
            evaluation['rank'+match[1]] = float(match[2])
        match = re.search(r'Full checkpoint saved: epoch=(\d+), successful_updates=(\d+), AMP skips=(\d+), scale=([\d.]+)', line)
        if match:
            e, updates, skips, scale = match.groups()
            checkpoints.append(dict(meta, epoch=int(e), successful_updates=int(updates), amp_skips=int(skips), scale=float(scale)))
        match = re.search(r'AMP skipped optimizer/prototype update: epoch=(\d+) iteration=(\d+) scale=([\d.]+) -> ([\d.]+)', line)
        if match:
            e, i, before, after = match.groups()
            overflows.append(dict(meta, epoch=int(e), iteration=int(i), before=float(before), after=float(after)))
        match = re.search(r'Resumed full training state from (.*): completed epoch=(\d+); next epoch=(\d+); AMP scale=([\d.]+)', line)
        if match:
            resumes.append(dict(meta, path=match[1], completed_epoch=int(match[2]), next_epoch=int(match[3]), scale=float(match[4])))
        match = re.search(r'Epoch (\d+) done. Time per batch: ([\d.]+)\[s\] Total: ([\d.]+)\[s\]', line)
        if match:
            timings.append(dict(meta, epoch=int(match[1]), per_batch=float(match[2]), train_seconds=float(match[3])))
    files.append(dict(name=path.name, bytes=path.stat().st_size, lines=len(lines),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(), first=lines[0][:23], last=lines[-1][:23],
                      warnings=[dict(line=i+1,text=line) for i,line in enumerate(lines) if re.search(r'WARNING|ERROR|Traceback|\bnan\b|\binf\b',line,re.I)]))

# The two files must be a non-overlapping epoch sequence, not independent experiments.
assert [r['epoch'] for r in checkpoints] == list(range(1,121))
assert [r['epoch'] for r in evaluations] == list(range(10,121,10))
assert all(len(r['pairs'])==9 and len(r['ag'])==2 for r in evaluations)
assert all(all(math.isfinite(v) for v in (r['loss'],r['accuracy'],r['lr'])) for r in training)
assert len(overflows) == checkpoints[-1]['amp_skips']
assert max(abs(r['loss_rounding_residual']) for r in training) < .003
assert all(abs(r['alpha']-.2*math.tanh(r['raw_alpha']/.2))<2e-6 and
           abs(r['beta']-.2*math.tanh(r['raw_beta']/.2))<2e-6 for r in qg)
for previous, current in zip(checkpoints,checkpoints[1:]):
    current['actual_batches'] = (current['successful_updates']+current['amp_skips'] -
                                 previous['successful_updates']-previous['amp_skips'])
checkpoints[0]['actual_batches'] = checkpoints[0]['successful_updates']+checkpoints[0]['amp_skips']
last_rows = {r['epoch']:r for r in training}
final_qg = [r for r in qg if r['epoch']==120 and r['iteration']==100]
assert len(final_qg)==12

write_csv('training.csv',training)
write_csv('qg_layers.csv',qg)
write_csv('checkpoints.csv',checkpoints)
write_csv('amp_overflows.csv',overflows)
write_csv('evaluation_overall.csv',[{k:v for k,v in r.items() if k not in ('pairs','ag')} for r in evaluations])
write_csv('evaluation_pairs.csv',[dict(epoch=r['epoch'],direction=key,**value) for r in evaluations for key,value in r['pairs'].items()])
write_csv('evaluation_ag.csv',[dict(epoch=r['epoch'],direction=key,**value) for r in evaluations for key,value in r['ag'].items()])
write_csv('qg_epoch120.csv',final_qg)

summary = dict(files=files, training_records=len(training), qg_records=len(qg), completed_epochs=120,
               missing_qg_snapshots=[dict(epoch=r['epoch'],iteration=r['iteration'],count=sum(q['epoch']==r['epoch'] and q['iteration']==r['iteration'] for q in qg))
                                     for r in training if sum(q['epoch']==r['epoch'] and q['iteration']==r['iteration'] for q in qg)!=12],
               resumes=resumes, amp_overflows=overflows, final_checkpoint=checkpoints[-1],
               skip_fraction=checkpoints[-1]['amp_skips']/(checkpoints[-1]['successful_updates']+checkpoints[-1]['amp_skips']),
               final_evaluation=evaluations[-1], final_training=last_rows[120], final_qg=final_qg,
               best_overall={metric:max(evaluations,key=lambda r:r[metric])['epoch'] for metric in ('map','rank1','rank5','rank10')},
               training_seconds=sum(r['train_seconds'] for r in timings),
               checkpoint_save_seconds=[(datetime.strptime(c['time'],'%Y-%m-%d %H:%M:%S,%f')-datetime.strptime(t['time'],'%Y-%m-%d %H:%M:%S,%f')).total_seconds()
                                        for c in checkpoints for t in timings if c['epoch']==t['epoch'] and c['epoch']%10],
               loss_rounding_residual_max=max(abs(r['loss_rounding_residual']) for r in training))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print('Files:',[(r['name'],r['lines']) for r in files])
print('Records:',len(training),len(qg),'incomplete QG:',summary['missing_qg_snapshots'])
print('Resume:',resumes)
print('AMP:',overflows)
print('Overall:',[(r['epoch'],r['rank1'],r['map'],r['rank5'],r['rank10']) for r in evaluations])
print('Final training:',last_rows[120])
print('Final qg:',[(r['layer'],r['alpha'],r['beta'],r['ratio'],round(r['alpha_derivative'],4)) for r in final_qg])
print('Time:',summary['training_seconds'],'checkpoint mean:',statistics.mean(summary['checkpoint_save_seconds']))

# Cross-check the whole AMP growth/backoff history, including its resume boundary.
scale, growth, updates = 65536., 0, 0
events = {(r['epoch'],r['iteration']):r for r in overflows}
for checkpoint in checkpoints:
    for iteration in range(1, checkpoint['actual_batches']+1):
        event = events.get((checkpoint['epoch'],iteration))
        if event:
            assert scale == event['before']
            scale *= .5
            growth = 0
            assert scale == event['after']
        else:
            updates += 1
            growth += 1
            if growth == 2000:
                scale *= 2
                growth = 0
    assert scale == checkpoint['scale'], checkpoint
    assert updates == checkpoint['successful_updates']

# Verify all logged LRs against the unchanged scheduler at the actual base LR.
import sys
sys.path.insert(0,str(ROOT/'WHU'))
import torch
from config import cfg
from solver.scheduler_factory import create_scheduler
config = cfg.clone()
config.merge_from_file(str(ROOT/'WHU/configs/UAD.yml'))
config.SOLVER.BASE_LR = .032
optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))],lr=.032)
scheduler = create_scheduler(config,optimizer)
assert all(float('{:.2e}'.format(scheduler._get_lr(r['epoch'])[0]))==r['lr'] for r in training)

# Values transcribed from visually checked WHU paper Tables 3/4/6, not a matched rerun.
paper_pairs = {
    '1_1':(40.47,26.83,5.36),'1_2':(26.56,16.52,2.43),'1_3':(8.10,6.00,1.11),
    '2_1':(25.53,16.58,3.10),'2_2':(29.79,17.93,2.64),'2_3':(7.78,5.87,1.11),
    '3_1':(9.04,6.32,1.36),'3_2':(8.57,5.69,1.02),'3_3':(20.98,14.00,2.35)}
comparison=[]
for direction,reference in paper_pairs.items():
    value = evaluations[-1]['pairs'][direction]
    result = dict(direction=direction)
    for metric,ref in zip(('rank1','map','minp'),reference):
        result[metric] = value[metric]
        result['paper_'+metric] = ref
        result['delta_'+metric] = round(value[metric]-ref,2)
    comparison.append(result)
write_csv('paper_reference_comparison.csv',comparison)
summary.update(amp_growth_history_verified=True,lr_history_verified=True,
               paper_comparison=comparison, nonfinite_qg_records=sum(
                   not all(math.isfinite(r[k]) for k in ('alpha','beta','ratio')) for r in qg))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')

# Scientific plots are standalone PNG/PDF artifacts with explicit units/scope.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2,3,figsize=(16,9),layout='constrained')
epochs = [r['epoch'] for r in evaluations]
for metric,label in [('rank1','Rank-1'),('rank5','Rank-5'),('rank10','Rank-10'),('map','mAP')]:
    axes[0,0].plot(epochs,[r[metric] for r in evaluations],marker='o',ms=3,label=label)
axes[0,0].set(title='Overall retrieval (full WHU-MARS-1000)',xlabel='Epoch',ylabel='Score (%)')
axes[0,0].axvline(40,color='gray',ls=':',label='Resume after epoch 40')
axes[0,0].legend(fontsize=8,ncol=2)
endpoints = list(last_rows.values())
for key,label in [('loss','Total'),('ce','CE'),('wrt','WRT'),('pca','ProCA (unweighted)'),('gpd','GPD')]:
    axes[0,1].plot([r['epoch'] for r in endpoints],[r[key] for r in endpoints],label=label)
axes[0,1].set(title='Rank-0 running means at iteration 100',xlabel='Epoch',ylabel='Loss',yscale='log')
axes[0,1].legend(fontsize=8)
axes[0,2].plot([r['epoch'] for r in endpoints],[r['accuracy']*100 for r in endpoints],label='Training accuracy')
axes[0,2].set(title='Training fit and learning rate',xlabel='Epoch',ylabel='Accuracy (%)')
twin=axes[0,2].twinx()
twin.plot([r['epoch'] for r in endpoints],[r['lr'] for r in endpoints],color='tab:orange',label='Learning rate')
twin.set_ylabel('Learning rate',color='tab:orange')
for ax,key,title,vmin,vmax,cmap in [(axes[1,0],'alpha','Effective alpha',-.2,.2,'coolwarm'),
                                    (axes[1,1],'beta','Effective beta',-.2,.2,'coolwarm'),
                                    (axes[1,2],'ratio','Centered-logit modulation ratio R',0,.16,'viridis')]:
    matrix=np.array([[next(r[key] for r in qg if r['epoch']==e and r['iteration']==100 and r['layer']==layer)
                      for e in range(1,121)] for layer in range(1,13)])
    im=ax.imshow(matrix,aspect='auto',origin='lower',extent=[.5,120.5,.5,12.5],vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set(title=title+' (rank-0 snapshot)',xlabel='Epoch',ylabel='Layer',yticks=[1,3,6,9,12])
    ax.axvline(40.5,color='white',ls=':',lw=1)
    fig.colorbar(im,ax=ax,shrink=.85)
fig.savefig(OUT/'training_dashboard.png',dpi=170)
fig.savefig(OUT/'training_dashboard.pdf')
plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,metric,title in zip(axes,('rank1','map','minp'),('Rank-1','mAP','mINP')):
    matrix=np.array([r['delta_'+metric] for r in comparison]).reshape(3,3)
    bound=max(abs(matrix).max(),.1)
    im=ax.imshow(matrix,cmap='RdBu',vmin=-bound,vmax=bound)
    ax.set(title=title+' change vs paper (pp)',xticks=range(3),yticks=range(3),
           xticklabels=['RGB','NIR','TIR'],yticklabels=['RGB','NIR','TIR'],xlabel='Gallery',ylabel='Query')
    for i in range(3):
        for j in range(3):
            ax.text(j,i,'{:+.2f}'.format(matrix[i,j]),ha='center',va='center',
                    color='white' if abs(matrix[i,j])>.65*bound else 'black',fontsize=12)
    fig.colorbar(im,ax=ax,shrink=.7)
fig.suptitle('Paper reference only: not a matched UAD baseline rerun',fontsize=12)
fig.savefig(OUT/'paper_reference_delta.png',dpi=170)
plt.close(fig)
print('Verified complete AMP growth/backoff history and scheduler LRs; comparison/plots exported')
