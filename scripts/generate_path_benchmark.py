"""Generate the fixed 100-instance PA-HGTS evaluation set.

The set spans three cross-section families, three point-count scales and deterministic grid phases and is used by the six-method path benchmark.
"""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gppaf.geometry import build_graph

BASE = {
    'container': {
        'outer': [(0,0),(12,0),(12,8),(10,8),(10,3),(2,3),(2,8),(0,8)],
        'holes': [],
    },
    'multihole': {
        'outer': [(0,0),(12,0),(12,9),(0,9)],
        'holes': [[(3,2),(5,2),(5,4),(3,4)],[(7,5),(9,5),(9,7),(7,7)]],
    },
    'crosshole': {
        'outer': [(0,0),(12,0),(12,9),(0,9)],
        'holes': [[(5,2),(7,2),(7,3.5),(8.5,3.5),(8.5,5.5),(7,5.5),(7,7),(5,7),(5,5.5),(3.5,5.5),(3.5,3.5),(5,3.5)]],
    },
}

SETTING = {
    ('container','S'):(0.8,0.66), ('container','M'):(0.8,0.74), ('container','L'):(0.8,0.80),
    ('multihole','S'):(1.0,0.60), ('multihole','M'):(1.0,0.70), ('multihole','L'):(1.0,0.76),
    ('crosshole','S'):(1.0,0.64), ('crosshole','M'):(1.0,0.70), ('crosshole','L'):(1.0,0.80),
}

PHASES=[(0.00,0.00),(0.15,0.35),(0.25,0.50),(0.35,0.15),(0.45,0.70),(0.55,0.30),
        (0.65,0.85),(0.75,0.45),(0.85,0.10),(0.20,0.80),(0.60,0.60)]


def make_case(family,size,phase_no,serial):
    bead,fill=SETTING[(family,size)]
    geometry=BASE[family]
    phase=PHASES[phase_no % len(PHASES)]
    case=dict(id=f'{family}_{size}_{serial:03d}',group_id=f'{family}_{size}',
              outer=geometry['outer'],holes=geometry['holes'],bead_width=bead,
              fill_rate=fill,phase=phase,k_neighbors=12,max_scale=2.1)
    graph=build_graph(**{k:v for k,v in case.items() if k in {'outer','holes','bead_width','fill_rate','phase','k_neighbors','max_scale'}})
    case['start']=0;case['end']=len(graph.points)-1
    return case,len(graph.points)


def generate():
    cases=[];counts=[];serial=1
    for family in ('container','multihole','crosshole'):
        for size in ('S','M','L'):
            for phase_no in range(11):
                case,n=make_case(family,size,phase_no,serial)
                if not 55 <= n <= 96:
                    raise RuntimeError(f'Unexpected node count {n} for {case["id"]}')
                cases.append(case);counts.append(n);serial+=1
    # 99 + one additional independent phase configuration.
    case,n=make_case('multihole','M',3,serial)
    case['id']='multihole_M_100';case['phase']=[0.92,0.58]
    graph=build_graph(**{k:v for k,v in case.items() if k in {'outer','holes','bead_width','fill_rate','phase','k_neighbors','max_scale'}})
    case['end']=len(graph.points)-1
    cases.append(case);counts.append(len(graph.points))
    if len(cases)!=100: raise AssertionError(len(cases))
    return cases,counts


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('data/path_validation_100.json'))
    args=p.parse_args();cases,counts=generate();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(cases,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(instances=len(cases),min_nodes=min(counts),max_nodes=max(counts),mean_nodes=sum(counts)/len(counts))))
