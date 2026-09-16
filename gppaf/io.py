"""Portable JSON/CSV export and run provenance utilities."""
from pathlib import Path
import csv,hashlib,json,platform,time
import numpy as np
import torch


def jsonable(x):
    if isinstance(x,Path): return str(x)
    if isinstance(x,torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x): return None
    return x


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def environment():
    import scipy,shapely,sklearn
    return dict(python=platform.python_version(),system=platform.platform(),processor=platform.processor(),
                torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,shapely=shapely.__version__,
                sklearn=sklearn.__version__,cuda=torch.cuda.is_available(),torch_threads=torch.get_num_threads())


def fingerprint(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_path_csv(path,points):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['x_mm','y_mm']+(['z_mm'] if np.asarray(points).shape[1]==3 else []));writer.writerows(points)


def path_svg(path,region,points,title='Computed path'):
    """Dependency-free SVG preview for inspecting a generated path."""
    import html
    points=np.asarray(points);xmin,ymin,xmax,ymax=region.bounds
    width=xmax-xmin;height=ymax-ymin;scale=520/max(width,height)
    def coords(p): return ' '.join('%.3f,%.3f'%((x-xmin)*scale+30,(ymax-y)*scale+60) for x,y in np.asarray(p)[:,:2])
    rings=[list(region.exterior.coords)]+[list(r.coords) for r in region.interiors]
    shapes=''.join('<polygon points="%s" fill="%s" stroke="#596575"/>'%(coords(p),'#edf1f4' if i==0 else 'white') for i,p in enumerate(rings))
    body='<svg xmlns="http://www.w3.org/2000/svg" width="600" height="620"><rect width="100%%" height="100%%" fill="white"/><text x="30" y="30" font-family="sans-serif" font-size="18">%s</text>%s<polyline points="%s" fill="none" stroke="#126a92" stroke-width="2"/></svg>'%(html.escape(title),shapes,coords(points))
    Path(path).write_text(body,encoding='utf-8')
