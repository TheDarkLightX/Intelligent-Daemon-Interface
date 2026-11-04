"""Run a preregistered CPU campaign; every attempted configuration is retained."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).parent
# Validation selects the deployed model. Test and challenge labels remain untouched.
CONFIGS=[(32,2,0,'guided',80),(32,3,0,'guided',80),(64,3,0,'guided',80),(64,3,1,'guided',80),(64,3,2,'guided',80),(128,3,0,'guided',80),(128,3,1,'guided',80),(128,4,2,'guided',80),(256,3,0,'guided',60),(256,4,1,'guided',60),(256,6,2,'guided',60),(64,3,0,'random',80)]

def one(config):
    votes,depth,seed,wiring,epochs=config
    name=f'v{votes}-d{depth}-s{seed}-{wiring}'
    out=ROOT/'runs'/name
    args=[sys.executable,'-m','idi.layered_chat.train','--out',str(out),'--votes',str(votes),'--depth',str(depth),'--seed',str(seed),'--wiring',wiring,'--epochs',str(epochs),'--eval-every','5']
    started=time.monotonic()
    with (ROOT/'runs'/(name+'.log')).open('w') as log:
        p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    result=dict(name=name,returncode=p.returncode,seconds=round(time.monotonic()-started,2))
    if p.returncode==0:
        report=json.loads((out/'report.json').read_text());result.update(validation_accuracy=report['best_validation_accuracy'],bytes=report['model_bytes'],examples_seen=report['examples_seen'])
    print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    (ROOT/'runs').mkdir(exist_ok=True)
    (ROOT/'runs'/'plan.json').write_text(json.dumps(dict(configurations=CONFIGS,selection='highest validation accuracy, then smallest model; test held out',max_workers=3),indent=2)+'\n')
    started=time.monotonic();results=[]
    with concurrent.futures.ThreadPoolExecutor(3) as ex:
        for result in ex.map(one,CONFIGS):
            results.append(result);(ROOT/'runs'/'campaign.json').write_text(json.dumps(dict(seconds=time.monotonic()-started,results=results),indent=2)+'\n')
