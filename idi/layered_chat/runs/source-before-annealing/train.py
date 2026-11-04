"""Reproducible CPU search with validation-only selection and untouched test sets."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import re
import time
from collections import Counter
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from .corpus import dataset
from .network import Network

STOP=set('a an the is are was be to of for in on and or with me my i you it its can could would should do does how what why when which tell about please explain show give example simple terms step by beginner briefly algorithm lab learning project help understand studying need key outline main useful practice makes pitfalls watch intuitive account through'.split())

def tokens(text):
    words=[w for w in re.findall('[a-z0-9]+',text.lower()) if w not in STOP]
    features=set('w:'+w for w in words)
    features.update('b:'+a+'_'+b for a,b in zip(words,words[1:]))
    for w in words:
        if len(w)>=3:
            w='^'+w+'$'
            features.update('c:'+w[i:i+3] for i in range(len(w)-2))
    return features

def fit_encoder(rows,limit=1536,kind="frequency"):
    # Training data only, deterministic document-frequency selection.
    counts=Counter(t for r in rows for t in tokens(r['text']))
    if kind=="words-first":
        words=sorted(t for t in counts if t.startswith('w:'))
        extras=sorted((t for t in counts if not t.startswith('w:')),key=lambda t:(-counts[t],t))
        selected=words+extras[:max(0,limit-len(words))]
    else:
        selected=sorted(counts,key=lambda t:(-counts[t],t))[:limit]
    return sorted(selected)

def encode(rows,vocabulary):
    index={t:i for i,t in enumerate(vocabulary)}
    x=np.zeros((len(rows),len(vocabulary)),np.float32)
    for i,r in enumerate(rows):
        for t in tokens(r['text']):
            if t in index:x[i,index[t]]=1
    return x,np.array([r['label'] for r in rows],dtype=np.int64)

def wire_guided(net,x,y,classes,votes):
    """Learn positive feature associations on training data, then sample wiring.

    Later layers primarily connect within class groups, with 10% cross-group
    wires. Only the wiring is guided; gate functions are fitted by gradients.
    """
    rng=net.rng
    for li,layer in enumerate(net.layers):
        previous=x.shape[1] if li==0 else classes*votes
        for c in range(classes):
            sl=slice(c*votes,(c+1)*votes)
            if li==0:
                pos=x[y==c].mean(0);neg=x[y!=c].mean(0)
                weights=np.maximum(pos-neg,0)
                order=np.argsort(-weights,kind='stable')[:64]
                p=weights[order]+0.005;p/=p.sum()
                for k in range(2):layer.wires[k,sl]=rng.choice(order,votes,p=p)
            else:
                for k in range(2):
                    values=rng.integers(c*votes,(c+1)*votes,votes)
                    mask=rng.random(votes)<0.1
                    values[mask]=rng.integers(0,previous,mask.sum())
                    layer.wires[k,sl]=values
        layer.scatter=[sp.csr_matrix((np.ones(classes*votes,np.float32),(np.arange(classes*votes),w)),shape=(classes*votes,previous)) for w in layer.wires]

def frozen(net,features,cs):
    return dict(schema='idi/layered-chat/1',table_order='msb-00-01-10-11',inputs=len(features),features=features,classes=len(cs),ids=[c['id'] for c in cs],layers=[dict(a=l.wires[0].tolist(),b=l.wires[1].tolist(),f=l.logits.argmax(1).tolist()) for l in net.layers])

def scores(model,x,batch=256):
    results=[]
    for start in range(0,len(x),batch):
        h=x[start:start+batch].astype(np.uint8)
        for layer in model['layers']:
            a=h[:,layer['a']];b=h[:,layer['b']];f=np.array(layer['f'],np.uint8)
            h=(f>>(3-2*a-b))&1
        results.append(h.reshape(len(h),model['classes'],-1).sum(2,dtype=np.int32))
    return np.concatenate(results)

def run(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    cs,ds=dataset();features=fit_encoder(ds['train'],args.features,args.encoder)
    matrices={k:encode(v,features) for k,v in ds.items()}
    x,y=matrices['train'];vx,vy=matrices['validation']
    if args.pilot:
        rng=np.random.default_rng(args.seed)
        select=np.concatenate([rng.choice(np.flatnonzero(y==c),min(48,sum(y==c)),replace=False) for c in range(len(cs))])
        x,y=x[select],y[select]
    config=vars(args).copy();config.pop('out');config['classes']=len(cs);config['input_bits']=len(features)
    net=Network(len(features),[len(cs)*args.votes]*args.depth,'choice',2,args.seed)
    if args.wiring=='guided':wire_guided(net,x,y,len(cs),args.votes)
    config['source_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('train.py','network.py','corpus.py')}
    started=time.monotonic();history=[];best=-1;exposures=0
    (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for epoch in range(args.epochs):
        order=net.rng.permutation(len(y))
        loss_total=0
        for start in range(0,len(y),args.batch):
            ix=order[start:start+args.batch]
            loss,grads=net.loss_and_grads(x[ix],y[ix],len(cs),max(1,args.votes/10))
            for layer,grad in zip(net.layers,grads):layer.adam(grad,args.rate)
            loss_total+=float(loss)*len(ix);exposures+=len(ix)
        model=frozen(net,features,cs)
        if epoch==0 or (epoch+1)%args.eval_every==0 or epoch+1==args.epochs:
            s=scores(model,vx);acc=float((s.argmax(1)==vy).mean())
            row=dict(epoch=epoch+1,loss=loss_total/len(y),validation_accuracy=acc,seconds=round(time.monotonic()-started,3),examples_seen=exposures)
            history.append(row);print(json.dumps(row),flush=True)
            if acc>best:
                best=acc;model['training']=dict(config,selected_epoch=epoch+1,validation_accuracy=best)
                (root/'model.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
            (root/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    model=json.loads((root/'model.json').read_text())
    report=dict(config=config,seconds=round(time.monotonic()-started,3),examples_seen=exposures,selected_epoch=model['training']['selected_epoch'],best_validation_accuracy=best,model_sha256=hashlib.sha256((root/'model.json').read_bytes()).hexdigest(),model_bytes=(root/'model.json').stat().st_size,training_rows=len(y),split_rows={k:len(v) for k,v in ds.items()},data_sha256=hashlib.sha256(json.dumps(ds,sort_keys=True).encode()).hexdigest(),python=platform.python_version(),numpy=np.__version__,history=history)
    # No test/challenge score is used here; evaluate the selected campaign model separately.
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--votes',type=int,default=64);p.add_argument('--depth',type=int,default=3);p.add_argument('--seed',type=int,default=0);p.add_argument('--batch',type=int,default=256);p.add_argument('--rate',type=float,default=.04);p.add_argument('--eval-every',type=int,default=5);p.add_argument('--wiring',choices=['guided','random'],default='guided');p.add_argument('--pilot',action='store_true');p.add_argument('--encoder',choices=['frequency','words-first'],default='frequency');p.add_argument('--features',type=int,default=1536);run(p.parse_args())
