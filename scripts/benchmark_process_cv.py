"""Run the six-method process-design comparison on all saved grouped folds.

This driver uses the fold-specific TC-DSFNet checkpoints and split files created
by cross_validate.py. For every fold, all six methods are evaluated on the same
test rows with the same Best-of-M candidate budget. Results are aggregated from
executable per-fold outputs; no manuscript figure values are embedded here.
"""
from pathlib import Path
import argparse, json, subprocess, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gppaf.io import save_json

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/processed'))
    p.add_argument('--cv-run',type=Path,default=Path('runs/cross_validation'))
    p.add_argument('--output',type=Path,default=Path('runs/process_comparison'))
    p.add_argument('--epochs',type=int,default=100,help='baseline training epochs for cVAE/Affine-cINN')
    p.add_argument('--candidates',type=int,default=256)
    p.add_argument('--test-limit',type=int,default=0,help='Optional per-domain test-row limit for an execution check; 0 uses every held-out row')
    p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    fold_records=[]
    for fold in range(5):
        fold_run=args.cv_run/f'fold_{fold}'
        if not (fold_run/'model.pt').exists():
            raise FileNotFoundError(f'Missing {fold_run}/model.pt; run cross_validate.py first')
        out=args.output/f'fold_{fold}'
        cmd=[sys.executable,str(Path(__file__).with_name('benchmark_process.py')),
             '--data',str(args.data),'--run',str(fold_run),'--output',str(out),
             '--epochs',str(args.epochs),'--candidates',str(args.candidates),'--test-limit',str(args.test_limit),'--seed',str(args.seed+fold)]
        subprocess.run(cmd,check=True)
        fold_records.append(json.loads((out/'results.json').read_text()))
    methods=['RSM','SVR','RF','cVAE','Affine-cINN','TC-DSFNet']
    aggregate={}
    for method in methods:
        aggregate[method]={}
        for domain in ['single','multi']:
            vals=[]
            for fold in fold_records:
                row=next(r for r in fold['records'] if r['method']==method and r['domain']==domain)
                vals.append(row['best_of_m_mae_percent'])
            aggregate[method][domain]={'mean_percent':float(np.mean(vals)),'std_across_folds':float(np.std(vals,ddof=1)),'fold_values':vals}
        aggregate[method]['mean_two_domains_percent']=float(np.mean([aggregate[method]['single']['mean_percent'],aggregate[method]['multi']['mean_percent']]))
    tc=aggregate['TC-DSFNet']['mean_two_domains_percent']
    comparisons={m:(aggregate[m]['mean_two_domains_percent']-tc)/aggregate[m]['mean_two_domains_percent']*100 for m in methods if m!='TC-DSFNet'}
    save_json(args.output/'summary.json',dict(aggregate=aggregate,relative_reduction_vs_each_comparison_percent=comparisons,
        protocol=dict(folds=5,split='same saved grouped folds for all methods',candidates=args.candidates,baseline_epochs=args.epochs,test_limit=args.test_limit,seed=args.seed),
        note='All values in this file are computed from executable fold outputs.'))
