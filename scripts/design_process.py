"""Generate and rank bounded process candidates from a saved TC-DSFNet run."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gppaf.process import TCDSFNet,design
from gppaf.io import save_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--domain',choices=['single','multi'],required=True)
    p.add_argument('--width',type=float,required=True);p.add_argument('--height',type=float,required=True)
    p.add_argument('--candidates',type=int,default=128);p.add_argument('--seed',type=int,default=123)
    p.add_argument('--output',type=Path,default=Path('runs/design/result.json'));args=p.parse_args()
    torch.set_num_threads(1);checkpoint=torch.load(str(args.run/'model.pt'),map_location='cpu')
    model=TCDSFNet(**checkpoint['config']);model.load_state_dict(checkpoint['state_dict']);model.eval()
    calibration=json.loads((args.run/'calibration.json').read_text(encoding='utf-8'))[args.domain]
    target=torch.tensor([[args.width,args.height]],dtype=torch.float32)
    result=design(model,target,args.domain,calibration,args.candidates,args.seed)
    with torch.no_grad(): prediction=model.predict_geometry(model.normalize(result['selected'],args.domain),args.domain)
    save_json(args.output,dict(domain=args.domain,target=target,selected_geometry_prediction=prediction,**result))
    print(str(args.output))
