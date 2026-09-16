"""Reviewer-oriented checks for manuscript/code/data alignment.

The default invocation verifies the packaged data and reads the archived,
version-labelled five-fold TC-DSFNet artifacts. Publication reference values are
reported in a separate field and are never used as inputs to executable metrics.
"""
from pathlib import Path
import argparse,csv,json,collections,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gppaf.io import fingerprint
from gppaf.geometry import build_graph


def jread(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def compact_process_cv(summary):
    aggregate=summary['aggregate']
    return {
        d:{
            'mean_best_of_m_percent':aggregate[d]['mean_best_of_m_percent'],
            'fold_best_of_m_percent':[f['evaluation'][d]['best_of_m_mae_percent'] for f in summary['folds']],
            'folds':len(summary['folds'])
        }
        for d in ('single','multi')
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--process-cv',type=Path,help='Optional replacement five-fold TC-DSFNet run directory')
    ap.add_argument('--process-comparison',type=Path,help='Optional six-method executable comparison directory')
    ap.add_argument('--path-results',type=Path,help='Optional six-method path benchmark directory')
    a=ap.parse_args(); root=a.root
    report={'checks':{},'publication_reference':{},'computed_results':{},'alignment':{}}

    # Source-table integrity and declared row counts.
    man=jread(root/'data/manifest.json'); ok=True; row_counts={}
    for e in man['files']:
        p=root/'data'/e['file']; ok &= p.exists() and fingerprint(p)==e['sha256']
        if p.exists():
            with p.open(encoding='utf-8-sig',newline='') as f: row_counts[Path(e['file']).name]=max(0,len(list(csv.reader(f)))-1)
    report['checks']['source_data_hashes']=bool(ok)
    report['computed_results']['source_table_rows']=row_counts

    # Layerwise completeness and row lineage.
    with (root/'data/processed/layers.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    groups=collections.defaultdict(list)
    for r in rows: groups[r['condition_id']].append(r)
    layer_complete=(len(rows)==1050 and len(groups)==105 and
        all(sorted(int(x['layer']) for x in g)==list(range(1,11)) for g in groups.values()))
    lineage_fields={'factor','x_variable','x_value','source_file','source_row'}
    lineage=bool(rows) and lineage_fields.issubset(rows[0]) and len({int(r['source_row']) for r in rows})==1050
    report['checks']['layerwise_105x10']=layer_complete
    report['checks']['layerwise_row_lineage']=lineage
    with (root/'data/processed/layer_condition_metadata.csv').open(encoding='utf-8') as f: meta=list(csv.DictReader(f))
    report['checks']['layer_condition_metadata_105']=len(meta)==105 and {m['condition_id'] for m in meta}==set(groups) and all(m['source_basis'].strip() for m in meta)
    report['computed_results']['layerwise']={'rows':len(rows),'conditions':len(groups),'layers_per_condition':10,'condition_metadata_rows':len(meta)}

    # Fixed 100-case evaluation set: validate geometry construction and range.
    cases=jread(root/'data/path_validation_100.json'); counts=[]; families=set(); phases=set()
    for c in cases:
        g=build_graph(**{k:v for k,v in c.items() if k in {'outer','holes','bead_width','fill_rate','phase','k_neighbors','max_scale','points'}})
        counts.append(len(g.points)); families.add(c['group_id'].split('_')[0]); phases.add(tuple(c['phase']))
    path_ok=(len(cases)==100 and counts and min(counts)>=55 and max(counts)<=96 and len(families)==3 and len(phases)>=10)
    report['checks']['path_validation_100']=path_ok
    report['computed_results']['path_validation_structure']={
        'instances':len(cases),'node_count_min':min(counts),'node_count_max':max(counts),
        'cross_section_families':len(families),'grid_phases':len(phases)
    }

    # Publication reference values are clearly separated from computed runs.
    manuscript_claims=root/'reference_results/manuscript_claims.json'
    if manuscript_claims.exists():
        report['publication_reference']['manuscript_claims']=jread(manuscript_claims)
    fig7=root/'reference_results/process_comparison/manuscript_reported_figure7.json'
    if fig7.exists():
        ref=jread(fig7); report['publication_reference']['figure7']=ref
        tc=ref['methods']['TC-DSFNet']['mean']
        best=ref['methods'][ref['best_comparison_method']]['mean']
        reduction=100.0*(best-tc)/best
        report['checks']['figure7_reference_arithmetic']=abs(reduction-ref['relative_reduction_percent'])<1e-9 and abs(round(reduction,1)-ref['reported_reduction_percent_rounded'])<1e-9
        report['checks']['figure7_protocol_M256']=int(ref.get('candidate_count_M',0))==256

    # Read the archived packaged fold artifacts by default, or an explicitly supplied fresh run.
    cv=a.process_cv or (root/'reference_results/cross_validation')
    if (cv/'summary.json').exists():
        summary=jread(cv/'summary.json')
        report['computed_results']['process_cv']=compact_process_cv(summary)
        report['computed_results']['process_cv']['artifact_implementation_version']=summary.get('artifact_implementation_version')
        report['computed_results']['process_cv']['artifact_status']=summary.get('artifact_status')
        report['checks']['process_cv_five_folds']=len(summary.get('folds',[]))==5
        if fig7.exists():
            ref=jread(fig7)['methods']['TC-DSFNet']
            run=report['computed_results']['process_cv']
            report['alignment']['tc_dsfnet_verification_vs_manuscript']={
                'single':{'publication_percent':ref['single'],'verification_run_percent':run['single']['mean_best_of_m_percent'],
                          'difference_percentage_points':run['single']['mean_best_of_m_percent']-ref['single']},
                'multi':{'publication_percent':ref['multi'],'verification_run_percent':run['multi']['mean_best_of_m_percent'],
                         'difference_percentage_points':run['multi']['mean_best_of_m_percent']-ref['multi']},
                'note':'Manuscript reference values are retained separately. The packaged five-fold artifacts are explicitly version-labelled and are not presented as outputs of the current implementation.'
            }
    else:
        report['checks']['process_cv_five_folds']=False

    if a.process_comparison and (a.process_comparison/'summary.json').exists():
        report['computed_results']['process_comparison']=jread(a.process_comparison/'summary.json')
    elif a.process_comparison and (a.process_comparison/'results.json').exists():
        report['computed_results']['process_comparison']=jread(a.process_comparison/'results.json')

    if a.path_results and (a.path_results/'summary.json').exists():
        s=jread(a.path_results/'summary.json')
        report['computed_results']['path_benchmark']=s.get('summary',s)
    else:
        example=root/'reference_results/path_benchmark_example/summary.json'
        if example.exists():
            s=jread(example)
            ex=s.get('summary',s)
            report['computed_results']['path_benchmark_example']=ex
            expected={'NN','ACO','G-beam','P-beam','R-GLS','PA-HGTS'}
            report['checks']['path_example_six_methods']=expected.issubset(set(ex))
            pa=ex.get('PA-HGTS',{})
            report['checks']['path_example_pa_hgts_execution']=pa.get('total_instances')==8 and pa.get('feasible_instances')==8
        else:
            report['checks']['path_example_six_methods']=False
            report['checks']['path_example_pa_hgts_execution']=False

    print(json.dumps(report,indent=2))
    if not all(report['checks'].values()): raise SystemExit(1)


if __name__=='__main__': main()
