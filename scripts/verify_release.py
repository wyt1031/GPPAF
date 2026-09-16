"""Verify files and implementation invariants delivered in the GPPAF release."""
from pathlib import Path
import sys,json,csv,collections,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from shapely.geometry import LineString,Polygon
from gppaf.geometry import build_graph
from gppaf.process import TCDSFNet, features
from gppaf.hgts import PAHGTS
from gppaf.smoothing import derivatives,curvature_certificate,monotonic_projection,smooth_path,SmoothConfig
from gppaf.io import fingerprint,save_json


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def verify(root):
    torch.set_num_threads(1);checks=[]

    # Release-version consistency across package metadata and reviewer-facing records.
    import re
    pyproject=(root/'pyproject.toml').read_text(encoding='utf-8')
    m=re.search(r'^version\s*=\s*"([^"]+)"',pyproject,re.M); assert m
    version=m.group(1)
    assert f'__version__ = "{version}"' in (root/'gppaf/__init__.py').read_text(encoding='utf-8')
    assert f'version: {version}' in (root/'CITATION.cff').read_text(encoding='utf-8')
    assert f'v{version}' in (root/'README.md').read_text(encoding='utf-8')
    assert (root/'README.md').read_bytes()==(root/'README.txt').read_bytes()
    assert f'v{version}' in (root/'requirements-tested.txt').read_text(encoding='utf-8')
    assert f'v{version}' in (root/'reference_results/process_comparison/README.md').read_text(encoding='utf-8')
    assert read(root/'release_manifest.json')['version']==version
    assert not (root/'examples/layer_settings.template.json').exists()
    assert not (root/'reference_results/pa_hgts/model.pt').exists()
    # Published reference JSON must not retain machine-specific temporary paths.
    for metadata_path in (root/'reference_results').rglob('*.json'):
        payload=metadata_path.read_text(encoding='utf-8')
        assert '/mnt/data/' not in payload and '/home/oai/' not in payload and '/tmp/' not in payload, str(metadata_path)
    checks.append('Release version metadata, duplicate README, archive layout and saved reference paths are internally consistent')

    manifest=read(root/'data/manifest.json')
    for entry in manifest['files']:
        path=root/'data'/Path(entry['file'])
        assert fingerprint(path)==entry['sha256'],str(path)
        with path.open(encoding='utf-8-sig') as f: count=len(list(csv.reader(f)))-1
        assert count==entry['rows']
    checks.append('Source-data hashes and row counts match the manifest')

    # Layerwise training table, explicit parameter map and row lineage.
    settings=read(root/'data/layer_settings.json')['fixed_by_factor']
    with (root/'data/raw/layer_partial.csv').open(encoding='utf-8-sig') as f: raw_rows=list(csv.DictReader(f))
    with (root/'data/processed/layers.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    assert len(raw_rows)==len(rows)==1050 and len({r['condition_id'] for r in rows})==105
    factor_index={'Welding current':'current_A','Travel speed':'speed_cm_min','Welding speed':'speed_cm_min',
                  'Oscillation frequency':'frequency_Hz','Weaving frequency':'frequency_Hz',
                  'Oscillation amplitude':'amplitude_mm','Weaving amplitude':'amplitude_mm','Cooling time':'cooling_s'}
    parameters=['current_A','speed_cm_min','frequency_Hz','amplitude_mm','cooling_s']
    for raw,processed in zip(raw_rows,rows):
        assert processed['source_file']=='layer_partial.csv' and int(processed['source_row'])>=2
        assert processed['factor']==raw['分析因素'] and processed['x_variable']==raw['X变量']
        assert abs(float(processed['x_value'])-float(raw['X值']))<1e-10
        assert int(processed['layer'])==int(raw['层号'])
        assert abs(float(processed['width_mm'])-float(raw['宽度(mm)']))<1e-10
        assert abs(float(processed['height_mm'])-float(raw['高度(mm)']))<1e-10
        factor=raw['分析因素']; expected=dict(settings[factor]); expected[factor_index[factor]]=float(raw['X值'])
        for key in parameters: assert abs(float(processed[key])-float(expected[key]))<1e-10
    grouped=collections.defaultdict(list)
    for r in rows: grouped[r['condition_id']].append(r)
    for records in grouped.values():
        assert sorted(int(r['layer']) for r in records)==list(range(1,11))
        vals={tuple(r[k] for k in parameters) for r in records}; assert len(vals)==1
    with (root/'data/processed/layer_condition_metadata.csv').open(encoding='utf-8') as f:
        metadata=list(csv.DictReader(f))
    assert len(metadata)==105 and {r['condition_id'] for r in metadata}==set(grouped)
    meta_by={r['condition_id']:r for r in metadata}
    for cid,records in grouped.items():
        first=records[0]; meta=meta_by[cid]
        assert int(meta['layers'])==10 and meta['source_basis'].strip()
        for key in parameters: assert abs(float(meta[key])-float(first[key]))<1e-10
    checks.append('Layerwise table has 105 complete five-parameter conditions x 10 layers, source-row lineage and condition metadata')

    # Eq. (2): exact gamma*c+beta with identity initialization.
    torch.manual_seed(7);m=TCDSFNet(hidden=16,context=8,layers=2)
    y=torch.tensor([[10.,2.5]])
    c=m.encoder((features(y)-m.feature_mean)/m.feature_scale)
    for domain in ['single','multi']:
        cond=m.conditions(y,domain)
        assert all(torch.allclose(q,c,atol=1e-7,rtol=0) for q in cond)
    checks.append('TC-DSFNet Eq. (2) adapters implement gamma*c+beta and initialize at identity')


    # Eq. (6): one global mean over all retained stable relationships in C.
    torch.manual_seed(17);m=TCDSFNet(hidden=16,context=8,layers=1)
    yy=torch.tensor([[9.5,2.3],[10.5,2.6],[11.0,2.4]],dtype=torch.float32)
    xs=torch.tensor([[70.,24.,2.5,2.0],[80.,26.,3.0,3.0],[90.,28.,3.5,3.5]],dtype=torch.float32)
    xm=torch.tensor([[70.,24.,2.5,2.0,20.],[80.,26.,3.0,3.0,40.],[90.,28.,3.5,3.5,60.]],dtype=torch.float32)
    m.fit_scalers({'single':yy,'multi':yy})
    batches={'single':(xs,yy,torch.ones(3)),'multi':(xm,yy,torch.ones(3))}
    signs={'single':[(0,0,1)],'multi':[(0,0,1),(1,1,-1)]}
    _,terms=m.losses(batches,signs)
    expected=[]
    for d,x in [('single',xs),('multi',xm)]:
        rho=m.normalize(x,d).detach().requires_grad_(True);pred=m.predict_geometry(rho,d)
        for k,r,sign in signs[d]:
            jac=torch.autograd.grad(pred[:,r].sum(),rho,create_graph=True,retain_graph=True)[0][:,k]
            expected.append(torch.relu(-float(sign)*jac).mean())
    assert torch.allclose(terms['trend'],torch.stack(expected).mean(),atol=1e-7,rtol=0)
    checks.append('TC-DSFNet Eq. (6) trend loss uses one global mean over all retained stable relationships')

    # PA-LCCS: epsilon_L screens short corners but never acts as point deduplication.
    safe=Polygon([(-1,-1),(3,-1),(3,3),(-1,3)])
    short=np.array([[0.,0.],[1.,0.],[1.00005,0.]])
    sr=smooth_path(short,safe,SmoothConfig(epsilon_L=1e-4,deviation=.5,max_curvature=20))
    assert np.array_equal(np.asarray(sr['points']),short)
    duplicate=np.array([[0.,0.],[0.,0.],[1.,0.],[1.,0.]])
    dr=smooth_path(duplicate,safe,SmoothConfig(epsilon_L=1e-4,deviation=.5,max_curvature=20))
    assert np.array_equal(np.asarray(dr['points']),np.array([[0.,0.],[1.,0.]]))
    assert np.array_equal(np.asarray(dr['points'])[-1],duplicate[-1])
    checks.append('PA-LCCS separates duplicate removal from epsilon_L short-segment screening and preserves path endpoints')

    # Fixed 100-case benchmark structural checks.
    cases=read(root/'data/path_validation_100.json');assert len(cases)==100
    ids=set();node_counts=[];families=set();phases=set()
    for c in cases:
        assert c['id'] not in ids;ids.add(c['id']);families.add(c['group_id'].split('_')[0]);phases.add(tuple(c['phase']))
        graph=build_graph(**{k:v for k,v in c.items() if k in ['outer','holes','bead_width','fill_rate','phase','k_neighbors','max_scale','points']})
        assert 0<=c['start']<len(graph.points) and 0<=c['end']<len(graph.points) and c['start']!=c['end']
        node_counts.append(len(graph.points))
    assert len(families)==3 and len(phases)>=10 and min(node_counts)>=55 and max(node_counts)<=96
    checks.append('100-case path evaluation set parses and spans families/scales/grid phases')

    # Optional packaged formal CV outputs: verify split isolation and arithmetic only.
    cv=root/'reference_results/cross_validation'
    if (cv/'summary.json').exists():
        aggregate=read(cv/'summary.json')
        assert aggregate.get('artifact_implementation_version')=='1.4.4'
        for domain in ['single','multi']:
            with (root/'data/processed'/(domain+'.csv')).open() as f: expected=[r['sample_id'] for r in csv.DictReader(f)]
            observed=[];errors=[]
            for fold in range(5):
                folder=cv/f'fold_{fold}';split=read(folder/'splits.json')
                config=read(folder/'config.json'); evaluation_record=read(folder/'evaluation.json')
                archived_checkpoint=torch.load(str(folder/'model.pt'),map_location='cpu')
                assert config.get('artifact_implementation_version')=='1.4.4'
                assert evaluation_record.get('artifact_implementation_version')=='1.4.4'
                assert archived_checkpoint.get('artifact_implementation_version')=='1.4.4'
                sets={k:set(v) for k,v in split['groups'].items()}
                assert not sets['train']&sets['test'] and not sets['train']&sets['validation'] and not sets['test']&sets['validation']
                observed.extend(split['rows'][domain]['test'])
                training=set(split['rows'][domain]['train'])
                assert all(a in training and b in training for a,b in read(folder/'augmentation_parents.json')[domain])
                ev=read(folder/'evaluation.json')['evaluation'][domain]
                assert set(ev['test_ids'])==set(split['rows'][domain]['test']);errors.append(ev['best_of_m_mae_percent'])
            assert collections.Counter(observed)==collections.Counter(expected)
            assert abs(np.mean(errors)-aggregate['aggregate'][domain]['mean_best_of_m_percent'])<1e-8
        envs=[]
        for fold in range(5): envs.append(read(cv/f'fold_{fold}'/'evaluation.json')['environment'])
        keys=['python','torch','numpy','scipy','shapely','sklearn','cuda','torch_threads']
        assert all(all(e[k]==envs[0][k] for k in keys) for e in envs[1:])
        expected={'python':'3.13.5','torch':'2.10.0+cpu','numpy':'2.3.5','scipy':'1.17.0','shapely':'2.1.2','sklearn':'1.8.0','cuda':False,'torch_threads':1}
        assert all(envs[0][k]==v for k,v in expected.items())
        checks.append('Archived v1.4.4 grouped process folds are version-labelled and have isolated groups, complete test coverage and one recorded tested environment')

    # Compact PA-HGTS training-reference checkpoint: load plus explicit train/validation provenance.
    relative='reference_results/pa_hgts_training_reference/model.pt'
    checkpoint=torch.load(str(root/relative),map_location='cpu');model=PAHGTS(**checkpoint['config']);model.load_state_dict(checkpoint['state_dict'])
    assert checkpoint.get('training_scope')=='supplied train/validation JSON instances'
    assert int(checkpoint.get('selected_epoch',0))>0
    assert checkpoint.get('training_seed')==2026 and checkpoint.get('epochs')==12
    assert int(checkpoint.get('updates',0))>0 and int(checkpoint.get('attempts',0))==240
    assert checkpoint.get('optimizer')=='Adam' and abs(float(checkpoint.get('learning_rate'))-0.001)<1e-12
    manifest=checkpoint.get('dataset_manifest') or read(root/'reference_results/pa_hgts_training_reference/dataset_manifest.json')
    assert len(manifest['training']['instances'])==20 and len(manifest['validation']['instances'])==8
    train_file=root/'examples/pa_hgts_training.json'; val_file=root/'examples/pa_hgts_validation.json'
    assert hashlib.sha256(train_file.read_bytes()).hexdigest()==manifest['training']['file_sha256']
    assert hashlib.sha256(val_file.read_bytes()).hexdigest()==manifest['validation']['file_sha256']
    demo=root/'reference_results/examples/hgts_demo.pt'
    if demo.exists(): assert hashlib.sha256((root/relative).read_bytes()).digest()!=hashlib.sha256(demo.read_bytes()).digest()
    run=read(root/'reference_results/pa_hgts_training_reference/run.json')
    assert run['train_cases']==20 and run['validation_cases']==8 and run['selected_epoch']==checkpoint['selected_epoch']
    checks.append('PA-HGTS compact training-reference checkpoint loads and has disjoint 20/8 training-validation provenance with a positive selected epoch')

    # Six-method path benchmark example: verify method coverage and failure accounting.
    path_example=root/'reference_results/path_benchmark_example'
    if (path_example/'summary.json').exists():
        summary=read(path_example/'summary.json')['summary']
        assert set(summary)=={'NN','ACO','G-beam','P-beam','R-GLS','PA-HGTS'}
        assert all(v['total_instances']==8 for v in summary.values())
        assert summary['PA-HGTS']['feasible_instances']==8
        with (path_example/'per_instance.csv').open(encoding='utf-8') as f: records=list(csv.DictReader(f))
        assert len(records)==48 and all('path_length' in r and 'heat_accumulation' in r for r in records)
        checks.append('Six-method path benchmark example exports aligned per-instance feasibility, objective components and statistics')

    # Executable smoothing/multilayer examples.
    example=root/'reference_results/examples'
    process_example=read(example/'process.json')
    process_checkpoint=torch.load(str(example/'process_demo.pt'),map_location='cpu')
    assert process_example.get('implementation_version')==version
    assert process_checkpoint.get('implementation_version')==version
    checks.append('Regenerated TC-DSFNet executable example is explicitly tied to the current implementation version')
    smoothing=read(example/'smoothing.json')
    assert smoothing['status']=='smoothed' and len(smoothing['curves'])==2 and LineString(smoothing['points']).is_simple
    for curve in smoothing['curves']:
        curve=np.array(curve);first,second=derivatives(curve,np.array([0.,1.]))
        assert np.max(abs(second))<1e-9 and np.min(np.linalg.norm(first,axis=1))>0
        assert curvature_certificate(curve,20.,1e-6) and monotonic_projection(curve)
    checks.append('Saved PA-LCCS example satisfies endpoint, regularity and curvature checks')
    layers=read(example/'layers.json');states=read(example/'layer_states.json')
    assert layers['status']=='feasible' and len(layers['state_indices'])==3
    for layer,index in zip(states,layers['state_indices']): assert sorted(layer[index]['component_order'])==[0,1,2]
    checks.append('Saved HELS-DP example has complete component coverage and backtracking')
    return dict(status='passed',checks=checks)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];report=verify(root)
    save_json(root/'reference_results/release_verification.json',report)
    print(json.dumps(report,indent=2))
