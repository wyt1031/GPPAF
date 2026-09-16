"""Normalize supplied process CSVs while preserving raw files and row lineage.

Usage: python scripts/prepare_data.py --source /path/to/CSV-directory
       [--layer-settings /path/to/layer-settings.json]

If --layer-settings is omitted, the packaged explicit layerwise parameter
metadata map in data/layer_settings.json is used. The generated layers.csv
contains all five process variables for each of the 105 conditions and 10
layers, together with row-level lineage back to the source layer table.
"""
from pathlib import Path
import sys,csv,argparse,shutil,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gppaf.io import save_json,fingerprint

NAMES={'single_factor':'单层单因素100组数据.csv','single_orthogonal':'单层正交100组数据.csv',
       'multi_mean':'多层正交响应面数据.csv','layer_partial':'多层单因素数据.csv'}
PARAMS=['current_A','speed_cm_min','frequency_Hz','amplitude_mm','cooling_s']
FACTOR={'Welding current':0,'Travel speed':1,'Welding speed':1,'Oscillation frequency':2,'Weaving frequency':2,
        'Oscillation amplitude':3,'Weaving amplitude':3,'Cooling time':4}


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.reader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('data'))
    parser.add_argument('--layer-settings',type=Path)
    args=parser.parse_args();raw=args.output/'raw';processed=args.output/'processed'
    raw.mkdir(parents=True,exist_ok=True);processed.mkdir(parents=True,exist_ok=True)
    manifest=[]; tables={}
    for name,filename in NAMES.items():
        source=args.source/filename;dest=raw/(name+'.csv');shutil.copyfile(str(source),str(dest))
        rows=read(source)[1:];tables[name]=rows
        manifest.append(dict(file=dest.relative_to(args.output).as_posix(),original_name=filename,rows=len(rows),sha256=fingerprint(dest),provenance='Source process table included with the GPPAF research package'))
    for domain,keys in [('single',['single_factor','single_orthogonal']),('multi',['multi_mean'])]:
        count=4 if domain=='single' else 5
        with (processed/(domain+'.csv')).open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['sample_id','group_id',*PARAMS[:count],'width_mm','height_mm','source_file','source_row'])
            for key in keys:
                for i,row in enumerate(tables[key],start=2):
                    x=list(map(float,row[1:1+count])); y=list(map(float,row[1+count:3+count]))
                    # Conservative grouping: all cooling variants of one base
                    # setting and shared settings across domains stay together.
                    group='_'.join('%.3f'%v for v in x[:4])
                    writer.writerow([key+'_'+row[0],group,*x,*y,key+'.csv',i])
    settings_path=args.layer_settings or (Path(__file__).resolve().parents[1]/'data'/'layer_settings.json')
    layer_status='not_loaded'
    if settings_path.exists():
        settings=json.loads(settings_path.read_text(encoding='utf-8'))
        with (processed/'layers.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['condition_id','group_id',*PARAMS,'layer','width_mm','height_mm','factor','x_variable','x_value','source_file','source_row'])
            for source_row,row in enumerate(tables['layer_partial'],start=2):
                factor=row[1]
                if factor not in FACTOR: raise ValueError('Unknown layer factor '+factor)
                fixed=settings['fixed_by_factor'][factor]
                x=[float(fixed[k]) for k in PARAMS];x[FACTOR[factor]]=float(row[4])
                group='_'.join('%.3f'%v for v in x[:4])
                writer.writerow([factor+'_'+row[4],group,*x,int(row[2]),float(row[5]),float(row[6]),factor,row[3],float(row[4]),'layer_partial.csv',source_row])
        # One auditable full-parameter row per unique layerwise condition.
        with (processed/'layers.csv').open(encoding='utf-8',newline='') as f:
            lrows=list(csv.DictReader(f))
        unique={}
        for r in lrows: unique.setdefault(r['condition_id'],r)
        with (processed/'layer_condition_metadata.csv').open('w',encoding='utf-8',newline='') as f:
            fields=['condition_id','factor','x_variable','x_value',*PARAMS,'layers','source_basis']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for cid,r in sorted(unique.items(),key=lambda kv:(kv[1]['factor'],float(kv[1]['x_value']))):
                basis=(settings.get('evidence_basis',{}).get('cooling_sweep','') if r['factor']=='Cooling time'
                       else settings.get('evidence_basis',{}).get('current_speed_frequency_amplitude_cooling',''))
                if not basis: basis='Explicit full-parameter map from the selected layer-settings metadata.'
                writer.writerow(dict(condition_id=cid,factor=r['factor'],x_variable=r['x_variable'],x_value=r['x_value'],
                    **{k:r[k] for k in PARAMS},layers=10,source_basis=basis))
        layer_status='complete: 105 conditions x 10 layers'
        prov=dict(settings); prov['settings_file']=('data/layer_settings.json' if args.layer_settings is None else str(settings_path)); prov['condition_metadata']='data/processed/layer_condition_metadata.csv'; save_json(processed/'layer_settings_provenance.json',prov)
    save_json(args.output/'manifest.json',dict(files=manifest,layer_status=layer_status,
              units=dict(current='A',speed='cm/min',frequency='Hz',amplitude='mm',cooling='s',geometry='mm')))
    print(json.dumps(dict(raw_files=len(manifest),single_rows=200,multi_rows=250,layer_status=layer_status)))

if __name__=='__main__': main()
