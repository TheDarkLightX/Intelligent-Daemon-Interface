"""Prepare an attributed OpenAssistant research subset and relevance examples.

The original source is Apache-2.0. This module never executes corpus text.
Selection uses released annotations and explicit size/scope filters, not a
claim that every retained answer has been independently fact-checked.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

REVISION='fdf72ae0827c1cda404aff25b6603abec9e3399b'
SOURCE=f'https://huggingface.co/datasets/OpenAssistant/oasst1/tree/{REVISION}'
STOP=set('a an the is are was were be been to of for in on and or with me my i you it its can could would should do does how what why when which tell about please explain show give this that these those as at by from your have has will just than if then not into'.split())
# Public demo stays with educational and ordinary conversational examples.
EXCLUDE=re.compile(r'\b(suicid\w*|self.harm|kill myself|porn\w*|sexual\w*|nude\w*|bomb\w*|weapon\w*|malware|ransomware|phishing|credit card|social security|diagnos\w*|prescription|dosage|chest pain|invest my|buy stock|legal advice)\b',re.I)

def terms(text):
    words=re.findall('[a-z0-9]+',text.lower())
    return [w[:-1] if w.endswith('s') and len(w)>4 and not w.endswith('ss') else w for w in words if w not in STOP]

def quality(m):return (m.get('labels') or {}).get('quality',{}).get('value',0)
def permitted(m):
    labels=m.get('labels') or {}
    return m.get('lang')=='en' and not m.get('deleted') and m.get('review_result') is True and not m.get('synthetic') and all(labels.get(k,{}).get('value',0)<=0 for k in ('pii','spam','not_appropriate','hate_speech','sexual_content')) and not EXCLUDE.search(m.get('text',''))

def prepare(path,out):
    records=[];counts=Counter();seen=set()
    with gzip.open(path,'rt') as stream:
        for line in stream:
            tree=json.loads(line);root=tree['prompt']
            def visit(prompt,context):
                counts['prompts_seen']+=1
                if prompt.get('role')!='prompter':return
                replies=[r for r in prompt.get('replies',[]) if permitted(r) and quality(r)>=.7]
                replies.sort(key=lambda r:(r.get('rank',999),-quality(r),r['message_id']))
                if permitted(prompt) and 15<=len(prompt['text'])<=400 and replies:
                    reply=replies[0];q=prompt['text'].strip();a=reply['text'].strip();key=' '.join(terms(q))
                    if 80<=len(a)<=2200 and len(terms(q))>=3 and key not in seen:
                        seen.add(key);records.append(dict(id=prompt['message_id'],answer_id=reply['message_id'],tree=tree['message_tree_id'],q=q,a=a,context=context[-900:],quality=round(quality(reply),4)))
                for reply in prompt.get('replies',[]):
                    for child in reply.get('replies',[]):visit(child,(context+' '+prompt['text'])[-900:])
            visit(root,'')
    records.sort(key=lambda r:(-r['quality'],r['id']));records=records[:5000]
    # Stable tree groups prevent branches of one conversation crossing splits.
    for r in records:
        b=int(hashlib.sha256(r['tree'].encode()).hexdigest()[:8],16)%10
        r['split']='test' if b==0 else 'validation' if b==1 else 'train'
    out.mkdir(parents=True,exist_ok=True)
    corpus=dict(schema='idi/openassistant-subset/1',source=SOURCE,license='Apache-2.0',attribution='OpenAssistant contributors / LAION, OASST1 (2023)',source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),changes='English human replies; annotation, length and scope filtering; best ranked qualifying reply per prompt; context limited to 900 characters; no user ids retained.',records=records)
    (out/'knowledge.json').write_text(json.dumps(corpus,ensure_ascii=False,separators=(',',':'))+'\n')
    report=dict(records=len(records),counts=dict(counts),splits=dict(Counter(r['split'] for r in records)),followup_records=sum(bool(r['context']) for r in records),source=SOURCE,source_sha256=corpus['source_sha256'],bytes=(out/'knowledge.json').stat().st_size,quality='Source annotations, not independent fact verification')
    (out/'knowledge-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return corpus

# Eight bounded quantities, eight threshold bits each. The learned second stage
# sees lexical relation features, not external embeddings or neural inference.
def relation(query,record):
    q=terms(query);d=terms(record['q']);a=terms(record['a']);qs=set(q);ds=set(d);aas=set(a);inter=len(qs&ds)
    big=lambda w:set(zip(w,w[1:]));qb=big(q);db=big(d)
    # Quantize entirely with integer arithmetic; matches the JS implementation.
    quantities=[8*inter//max(1,len(qs)),8*inter//max(1,len(ds)),8*len(qs&aas)//max(1,len(qs)),8*len(qb&db)//max(1,len(qb)),min(inter,8),8*len(qs&ds)//max(1,len(qs|ds)),8*min(len(q),len(d))//max(1,len(q),len(d)),int(q==d)*8]
    return [int(v>=t) for v in quantities for t in range(1,9)]

def pairs(corpus,out):
    rs=corpus['records'];rng=np.random.default_rng(20261104)
    vectorizer=TfidfVectorizer(tokenizer=terms,token_pattern=None,lowercase=False,max_features=20000)
    matrix=vectorizer.fit_transform(r['q'] for r in rs)
    bysplit={k:[] for k in ('train','validation','test')}
    for i,r in enumerate(rs):
        words=terms(r['q']); variants=[r['q'],' '.join(w for j,w in enumerate(words) if j%4!=2),' '.join(words[1:] or words)]
        neighbors=(matrix[i]@matrix.T).toarray()[0];neighbors[i]=-1
        hard=np.argsort(-neighbors,kind='stable')[:4].tolist()
        randoms=rng.integers(0,len(rs),2).tolist()
        for query in variants:
            if len(terms(query))<2:continue
            bysplit[r['split']].append(dict(x=relation(query,r),y=1,query_id=r['id'],candidate_id=r['id']))
            for j in hard+randoms:
                if j!=i:bysplit[r['split']].append(dict(x=relation(query,rs[j]),y=0,query_id=r['id'],candidate_id=rs[j]['id']))
    for split,rows in bysplit.items():
        np.savez_compressed(out/(split+'-pairs.npz'),x=np.array([r['x'] for r in rows],np.float32),y=np.array([r['y'] for r in rows],np.int64))
    (out/'pairs-report.json').write_text(json.dumps(dict(rows={k:len(v) for k,v in bysplit.items()},label_scope='Positive means original paired answer; other answers are weak negative labels and can also be relevant.',split='SHA256 conversation-tree ID; no tree spans splits',features='64 integer threshold bits; query/candidate lexical overlap'),indent=2)+'\n')
    print({k:len(v) for k,v in bysplit.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);corpus=prepare(a.source,out);pairs(corpus,out)
