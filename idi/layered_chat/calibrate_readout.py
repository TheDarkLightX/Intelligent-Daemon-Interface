"""Fit 62 integer vote offsets on training rows, keeping every Boolean gate fixed.

The small readout extends the uniform-vote classifier. It does not pad the table
count or fit on validation/test labels. A common offset keeps all exported
biases nonnegative without changing predictions or vote margins.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .corpus import dataset
from .train import encode, scores


def run(checkpoint, out):
    out.mkdir(parents=True, exist_ok=False)
    raw = checkpoint.read_bytes()
    model = json.loads(raw)
    if 'readout_bias' in model:
        raise ValueError('Calibrate a uniform-vote checkpoint exactly once')
    (out / 'base-model.json').write_bytes(raw)
    started = time.monotonic()
    _, data = dataset()
    x, y = encode(data['train'], model['features'])
    training_scores = scores(model, x)
    del x
    rows, counts = np.unique(np.c_[training_scores, y], axis=0, return_counts=True)
    features, labels = rows[:, :-1].astype(np.float64), rows[:, -1]
    weights = counts / counts.sum()
    temperature = 20.0
    penalty = 0.001

    def objective(offsets):
        logits = features / temperature + offsets
        normalizer = logsumexp(logits, axis=1)
        loss = np.dot(weights, normalizer - logits[np.arange(len(labels)), labels])
        probabilities = np.exp(logits - normalizer[:, None])
        probabilities[np.arange(len(labels)), labels] -= 1
        gradient = weights @ probabilities + 2 * penalty * offsets
        return float(loss + penalty * np.dot(offsets, offsets)), gradient

    fit = minimize(objective, np.zeros(model['classes']), jac=True, method='L-BFGS-B',
                   bounds=[(-32, 32)] * model['classes'], options={'maxiter':300,'ftol':1e-12,'gtol':1e-8})
    bias = np.rint(fit.x * temperature).astype(np.int32)
    bias -= bias.min()
    model['readout_bias'] = bias.tolist()
    model['training'] = {**model['training'], 'readout':'training-only integer class offsets'}
    vx, vy = encode(data['validation'], model['features'])
    before = scores(json.loads(raw), vx)
    after = scores(model, vx)
    assert np.array_equal(after, before + bias)
    report = dict(base_checkpoint_sha256=hashlib.sha256(raw).hexdigest(), training_rows=len(y),
                  unique_score_label_rows=len(labels), parameters=len(bias), temperature=temperature,
                  l2_penalty=penalty, optimizer_success=bool(fit.success), optimizer_message=str(fit.message),
                  iterations=int(fit.nit), objective=float(fit.fun),
                  raw_validation_accuracy=float((before.argmax(1)==vy).mean()),
                  best_validation_accuracy=float((after.argmax(1)==vy).mean()),
                  raw_training_accuracy=float((training_scores.argmax(1)==y).mean()),
                  calibrated_training_accuracy=float(((training_scores+bias).argmax(1)==y).mean()),
                  examples_seen=len(y), seconds=round(time.monotonic()-started,3),
                  source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
                                 for n in ('calibrate_readout.py','train.py','corpus.py','graph.py')},
                  scope='Train-only readout fit; validation used for architecture selection. One offset per class, rounded to integer votes. No test/challenge evaluation here.')
    model['training']['validation_accuracy'] = report['best_validation_accuracy']
    (out/'model.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
    report['model_sha256']=hashlib.sha256((out/'model.json').read_bytes()).hexdigest()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.checkpoint,args.out)
