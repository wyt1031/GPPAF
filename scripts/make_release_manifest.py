"""Create or verify the SHA-256 manifest for the GPPAF source release."""
from pathlib import Path
import argparse, hashlib, json

ROOT=Path(__file__).resolve().parents[1]
EXCLUDE_DIRS={'.git','.pytest_cache','__pycache__','.venv','venv','runs'}
EXCLUDE_FILES={'release_manifest.json'}

def sha256(path,chunk=1<<20):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(chunk),b''): h.update(block)
    return h.hexdigest()

def files(root):
    out=[]
    for p in root.rglob('*'):
        if not p.is_file(): continue
        rel=p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts): continue
        if rel.as_posix() in EXCLUDE_FILES: continue
        out.append(p)
    return sorted(out,key=lambda p:p.relative_to(root).as_posix())

def build(root):
    rows=[]
    for p in files(root):
        rows.append({'path':p.relative_to(root).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha256(p)})
    
    import re
    pyproject=(root/'pyproject.toml').read_text(encoding='utf-8')
    m=re.search(r'^version\s*=\s*"([^"]+)"',pyproject,re.M)
    version=m.group(1) if m else 'unknown'
    return {'name':'GPPAF research source code','version':version,'hash_algorithm':'SHA-256','file_count':len(rows),'files':rows}

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--check',action='store_true'); a=ap.parse_args()
    path=ROOT/'release_manifest.json'; current=build(ROOT)
    if a.check:
        expected=json.loads(path.read_text(encoding='utf-8'))
        if expected!=current: raise SystemExit('release_manifest.json does not match the current release tree')
        print(f"manifest ok: {current['file_count']} files")
    else:
        path.write_text(json.dumps(current,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(f"wrote {path}: {current['file_count']} files")
if __name__=='__main__': main()
