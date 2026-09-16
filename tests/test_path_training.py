"""Dataset isolation, failed-case accounting and held-out checkpoint selection."""
import argparse
import copy
import json
from pathlib import Path

import pytest
import torch

from gppaf.hgts import PAHGTS
from gppaf.io import save_json
from gppaf.path_data import load_instances, require_disjoint
from scripts.train_hgts import evaluate, run

ROOT = Path(__file__).resolve().parents[1]


def test_same_region_cannot_cross_splits_after_endpoint_and_ring_changes(tmp_path):
    records = json.loads((ROOT/'examples/path_training.json').read_text())
    _, train, _ = load_instances(ROOT/'examples/path_training.json')
    changed = copy.deepcopy(records[1])
    changed['outer'] = changed['outer'][2:] + changed['outer'][:2]
    changed['outer'].insert(1, [1., 2.5])  # An extra collinear polygon vertex.
    changed['holes'][0] = list(reversed(changed['holes'][0]))
    changed['points'] = list(reversed(changed['points']))
    changed['start'], changed['end'] = changed['end'], changed['start']
    changed['group_id'] = 'different-label-cannot-hide-identical-region'
    save_json(tmp_path/'duplicate.json', [changed])
    _, validation, _ = load_instances(tmp_path/'duplicate.json')
    with pytest.raises(ValueError, match='geometry_sha256'):
        require_disjoint(train, validation)


def test_declared_related_groups_and_invalid_endpoints(tmp_path):
    _, train, _ = load_instances(ROOT/'examples/path_training.json')
    records = json.loads((ROOT/'examples/path_validation.json').read_text())
    records[0]['group_id'] = train['instances'][0]['group_id']
    save_json(tmp_path/'related.json', records)
    _, validation, _ = load_instances(tmp_path/'related.json')
    with pytest.raises(ValueError, match='group_id'):
        require_disjoint(train, validation)
    records[0]['end'] = 999
    save_json(tmp_path/'invalid.json', records)
    with pytest.raises(ValueError, match='endpoints'):
        load_instances(tmp_path/'invalid.json')


def test_training_selects_validation_checkpoint_and_keeps_failures(tmp_path):
    args = argparse.Namespace(instances=ROOT/'examples/path_training.json',
        validation_instances=ROOT/'examples/path_validation.json',
        epochs=3, hidden=8, seed=2026, lr=.001, failure_penalty=1000., output=tmp_path/'run')
    run(args)
    report = json.loads((args.output/'run.json').read_text())
    history = json.loads((args.output/'validation_history.json').read_text())
    checkpoint = torch.load(str(args.output/'model.pt'), map_location='cpu')
    assert report['attempts'] == 6 and report['updates'] > 0
    assert len(history) == 4  # Includes the untrained epoch-zero reference.
    assert all(row['total'] == 2 and row['failures'] >= 1 for row in history)
    assert all(row['cases'][1]['path'] is None and row['cases'][1]['objective'] == 1000. for row in history)
    best = min(history, key=lambda row: (row['failures'], row['mean_penalized_objective']))
    assert report['selected_epoch'] == checkpoint['selected_epoch'] == best['epoch']
    model = PAHGTS(**checkpoint['config']); model.load_state_dict(checkpoint['state_dict'])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cases, _, _ = load_instances(args.validation_instances)
    actual = evaluate(model, cases, args.failure_penalty)
    assert model.training and all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert actual['cases'] == best['cases']
    assert actual['mean_penalized_objective'] == best['mean_penalized_objective']
    assert (args.output/'inputs/train.json').is_file()
    assert (args.output/'inputs/validation.json').is_file()
    assert (args.output/'last_model.pt').is_file()
