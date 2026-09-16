"""Load explicit path instances and keep related geometries in one split.

The exact polygon key is insensitive to ring starting vertex, ring direction,
hole order, point sampling and endpoint choices. Rotated, translated, scaled
or otherwise related specimens must additionally share an author-supplied
``group_id``; exact geometry matching cannot identify their provenance.
"""
import hashlib
import json
from pathlib import Path

from .geometry import build_graph


GRAPH_FIELDS = {'outer', 'holes', 'bead_width', 'fill_rate', 'phase',
                'k_neighbors', 'max_scale', 'points'}


def _ring_key(ring):
    points = [tuple(round(float(v), 9) for v in p) for p in ring.coords]
    if points[-1] == points[0]:
        points.pop()
    variants = []
    for order in (points, list(reversed(points))):
        smallest = min(order)
        for i, point in enumerate(order):
            if point == smallest:
                variants.append(tuple(order[i:] + order[:i]))
    return min(variants)


def geometry_key(graph):
    """Conservative exact-region group, independent of graph discretization."""
    region = graph.region.simplify(0., preserve_topology=True)
    value = (_ring_key(region.exterior),
             sorted(_ring_key(ring) for ring in region.interiors))
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def load_instances(path):
    """Read a nonempty JSON list using the same graph fields as plan_path.py.

    Endpoint indices refer to the built graph's y-then-x sorted distinct points.
    Multiple endpoint tasks from one geometry can coexist within a split.
    """
    path = Path(path)
    def reject_constant(value):
        raise ValueError('Non-finite JSON number: ' + value)
    records = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject_constant)
    if not isinstance(records, list) or not records:
        raise ValueError('Path dataset must be a nonempty JSON list: ' + str(path))
    cases, manifest, ids = [], [], set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError('Each instance must be an object')
        unknown = set(record) - GRAPH_FIELDS - {'start', 'end', 'id', 'group_id'}
        if unknown:
            raise ValueError('Unknown instance fields: ' + ', '.join(sorted(unknown)))
        case_id = record.get('id', '%s:%d' % (path.name, index))
        group_id = record.get('group_id')
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError('Instance IDs must be unique nonempty strings within each file')
        if group_id is not None and (not isinstance(group_id, str) or not group_id):
            raise ValueError('group_id must be a nonempty string when supplied')
        graph = build_graph(**{k: v for k, v in record.items() if k in GRAPH_FIELDS})
        start, end = record.get('start', 0), record.get('end', len(graph.points) - 1)
        if (type(start) is not int or type(end) is not int or
                not 0 <= start < len(graph.points) or not 0 <= end < len(graph.points) or start == end):
            raise ValueError('Invalid sorted-node endpoints for ' + case_id)
        ids.add(case_id)
        cases.append((graph, start, end))
        manifest.append(dict(id=case_id, group_id=group_id, geometry_sha256=geometry_key(graph),
                             nodes=len(graph.points), edges=len(graph.edge_index), start=start, end=end))
    return cases, dict(file_name=path.name, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       instances=manifest), records


def require_disjoint(train_manifest, validation_manifest):
    """Reject identical regions or declared shared specimen families across splits."""
    for field in ('geometry_sha256', 'group_id'):
        train = {item[field] for item in train_manifest['instances'] if item[field] is not None}
        validation = {item[field] for item in validation_manifest['instances'] if item[field] is not None}
        if train & validation:
            raise ValueError('Training/validation overlap in %s; keep related geometries in one split' % field)
