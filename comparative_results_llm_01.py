import json, glob, os, re, pandas as pd

def _first(d,*k):
    for i in k:
        if i in d and d[i] is not None:
            return d[i]
    return None

def _pull(obj,name):
    a={'accuracy':('accuracy','acc'),
       'precision':('precision','prec'),
       'recall':('recall','tpr'),
       'f1':('f1','f1_score')}
    v=_first(obj,*a[name])
    if v is None:
        for k in('metrics','eval','results'):
            if k in obj and isinstance(obj[k],dict):
                v=_first(obj[k],*a[name])
                if v is not None:break
    if v is None and 'classification_report' in obj:
        cr=obj['classification_report']
        if name=='accuracy':
            v=_first(cr,'accuracy')
        else:
            key='f1-score' if name=='f1' else name
            for agg in('weighted avg','macro avg'):
                if agg in cr and key in cr[agg]:
                    v=cr[agg][key]
                    break
    return v

records=[]
for p in glob.glob('experiments/**/results.json',recursive=True):
    with open(p) as f:
        d=json.load(f)
    bid=d.get('baseline_id') or re.search(r'experiments[/\\]([^/\\]+)',p).group(1)
    if not bid.startswith('bl_llm_01'):
        continue
    acc=_pull(d,'accuracy')
    prec=_pull(d,'precision')
    rec=_pull(d,'recall')
    f1=_pull(d,'f1')
    if None in(acc,prec,rec,f1):raise ValueError(bid)
    records.append({'baseline_id':bid,'accuracy':float(acc),'precision':float(prec),'recall':float(rec),'f1':float(f1)})

df=pd.DataFrame(records).sort_values('accuracy',ascending=False)
df.to_csv('llm01_results.csv',index=False)
print(df)
