import json, glob, os, re, pandas as pd

def _first(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _pull_metric(obj, name):
    aliases = {
        'accuracy':  ('accuracy', 'acc', 'accuracy_score'),
        'precision': ('precision', 'prec'),
        'recall':    ('recall', 'tpr'),
        'f1':        ('f1', 'f1_score', 'f1_macro')
    }
    v = _first(obj, *aliases[name])
    if v is not None:
        return v
    for k in ('metrics', 'eval', 'results'):
        if k in obj and isinstance(obj[k], dict):
            v = _first(obj[k], *aliases[name])
            if v is not None:
                return v
    if 'classification_report' in obj and isinstance(obj['classification_report'], dict):
        cr = obj['classification_report']
        if name == 'accuracy':
            v = _first(cr, 'accuracy')
            if v is not None:
                return v
        key = 'f1-score' if name == 'f1' else name
        for agg in ('weighted avg', 'macro avg'):
            if agg in cr and key in cr[agg]:
                return cr[agg][key]
    return None

records = []
for path in glob.glob('experiments/**/results.json', recursive=True):
    with open(path) as f:
        data = json.load(f)
    bid = data.get('baseline_id')
    if not bid:
        m = re.search(r'experiments[/\\]([^/\\]+)', path)
        bid = m.group(1) if m else os.path.basename(os.path.dirname(path))
    acc  = _pull_metric(data, 'accuracy')
    prec = _pull_metric(data, 'precision')
    rec  = _pull_metric(data, 'recall')
    f1   = _pull_metric(data, 'f1')
    if None in (acc, prec, rec, f1):
        raise ValueError(f'Missing metric for {bid}')
    records.append({'baseline_id': bid,
                    'accuracy_ood': float(acc),
                    'precision_ood': float(prec),
                    'recall_ood': float(rec),
                    'f1_ood': float(f1)})

ood_df = pd.DataFrame(records)

original_paper_results = {
    'bl_nlp_01': {'accuracy': 0.916,  'precision': 0.93,  'recall': 0.92,  'f1': 0.92},
    'bl_nlp_02': {'accuracy': 0.974,  'precision': 0.99,  'recall': 0.98,  'f1': 0.98},
    'bl_nn_01':  {'accuracy': 0.9982, 'precision': 0.9939,'recall': 0.978, 'f1': 0.9856},
    'bl_nn_02':  {'accuracy': 0.996,  'precision': 0.9937,'recall': 0.9951,'f1': 0.9944},
    'bl_llm_02': {'accuracy': 0.9861, 'precision': 0.9899,'recall': 0.9815,'f1': 0.9857},
}

paper_df = (pd.DataFrame.from_dict(original_paper_results, orient='index')
            .reset_index().rename(columns={'index': 'baseline_id'}))
paper_df = paper_df.rename(columns={c: f'{c}_paper' for c in ['accuracy', 'precision', 'recall', 'f1']})

merged = ood_df.merge(paper_df, on='baseline_id', how='inner')
for m in ['accuracy', 'precision', 'recall', 'f1']:
    merged[f'{m}_delta'] = merged[f'{m}_ood'] - merged[f'{m}_paper']

cols = ['baseline_id'] + [f'{m}_{s}' for m in ['accuracy', 'precision', 'recall', 'f1'] for s in ['paper', 'ood', 'delta']]
merged = merged[cols]
merged.to_csv('results_comparison.csv', index=False)
print(merged)
