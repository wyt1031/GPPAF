"""Run every grouped outer fold as an independent train/calibrate/test job."""
from pathlib import Path
import argparse,json,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gppaf.io import save_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--data',type=Path,default=Path('data/processed'));p.add_argument('--output',type=Path,default=Path('runs/cross_validation'))
    p.add_argument('--ablation',choices=['full','no_trend','no_layer'],default='full');p.add_argument('--candidates',type=int,default=256);args=p.parse_args();values=[]
    for fold in range(5):
        output=args.output/('fold_%d'%fold)
        subprocess.run([sys.executable,str(Path(__file__).with_name('train_process.py')),'--data',str(args.data),'--epochs',str(args.epochs),
                        '--fold',str(fold),'--output',str(output),'--ablation',args.ablation,'--candidates',str(args.candidates)],check=True)
        values.append(json.loads((output/'evaluation.json').read_text()))
    aggregate={d:dict(mean_best_of_m_percent=float(np.mean([v['evaluation'][d]['best_of_m_mae_percent'] for v in values])),
                     std_across_folds=float(np.std([v['evaluation'][d]['best_of_m_mae_percent'] for v in values],ddof=1))) for d in ['single','multi']}
    save_json(args.output/'summary.json',dict(folds=values,aggregate=aggregate,protocol=dict(folds=5,epochs=args.epochs,candidates=args.candidates,split='grouped',seed=2026)))
