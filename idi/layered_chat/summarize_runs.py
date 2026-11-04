"""Inventory completed, interrupted and unstarted work without counting padding."""
import hashlib
import json
from pathlib import Path


def run():
    root = Path(__file__).parent
    runs = root / 'runs'
    read = lambda p: json.loads(p.read_text())
    rows = []
    for directory in sorted(p for p in runs.iterdir() if p.is_dir()):
        config_path = directory / 'config.json'
        report_path = directory / 'report.json'
        history_path = directory / 'history.json'
        if not config_path.exists():
            continue
        config = read(config_path)
        report = read(report_path) if report_path.exists() else {}
        history = read(history_path) if history_path.exists() else []
        model_path = directory / 'model.json'
        model = read(model_path) if model_path.exists() else None
        row = dict(run=directory.name, stage='topic-gate-training',
                   status='completed' if report else 'partial',
                   table_count=config['classes'] * config['votes'] * config['depth'],
                   requested_epochs=config['epochs'], completed_logged_epochs=history[-1]['epoch'] if history else 0,
                   exposures=report.get('examples_seen', history[-1]['examples_seen'] if history else 0),
                   exposure_scope='exact completed run' if report else 'lower bound from last completed logged epoch',
                   validation_accuracy=report.get('best_validation_accuracy',model['training']['validation_accuracy'] if model else None),
                   selected_epoch=model['training']['selected_epoch'] if model else None,
                   model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest() if model else None,
                   encoder=config.get('encoder','frequency'), features=config['input_bits'], seed=config['seed'],
                   wiring=config['wiring'], depth=config['depth'])
        rows.append(row)
    named = {r['run'] for r in rows}
    for votes, depth, seed, wiring, epochs in read(runs/'plan.json')['configurations']:
        name=f'v{votes}-d{depth}-s{seed}-{wiring}'
        if name not in named:
            rows.append(dict(run=name,stage='topic-gate-training',status='no-completed-epoch' if (runs/(name+'.log')).exists() else 'not-started',table_count=62*votes*depth,requested_epochs=epochs,completed_logged_epochs=0,exposures=0))
    for directory in sorted(p for p in runs.iterdir() if p.is_dir()):
        report_path=directory/'report.json'
        if not report_path.exists() or directory.name in named:
            continue
        report=read(report_path)
        if directory.name.startswith('relevance'):
            rows.append(dict(run=directory.name,stage='relevance-gate-training',status='completed',table_count=report['tables'],completed_logged_epochs=report['history'][-1]['epoch'],exposures=report['examples_seen'],validation_balanced_accuracy=report['best_validation_balanced_accuracy']))
        elif directory.name.startswith('calibrated'):
            rows.append(dict(run=directory.name,stage='integer-readout-fitting',status='completed',training_rows=report['training_rows'],iterations=report['iterations'],parameters=report['parameters'],validation_accuracy=report['best_validation_accuracy'],exposures=0,scope='Rows and optimization iterations listed separately; not counted as new gate-training exposures.'))
        elif directory.name.startswith('ensemble-111104'):
            rows.append(dict(run=directory.name,stage='exact-vote-composition',status='completed',table_count=report['tables'],validation_accuracy=report['best_validation_accuracy'],exposures=0,scope='Member training is already counted, not counted again here.'))
    for attempt in sorted(runs.glob('*-oom-attempt/attempt.json')):
        rows.append(dict(run=attempt.parent.name,stage='topic-gate-training',status='resource-failure-before-first-metric',exposures=0,details=read(attempt)))
    report=dict(schema='idi/cpu-campaign/1',hardware=dict(cpu_quota_cores=8,memory_limit_bytes=8589934592,gpu=False),
                scope='Exposures count repeated presentations, not unique examples or dialogue turns. Partial runs receive credit only through their last logged completed epoch. Pilot data differ from the later corpus. Relevance reproduction repeats the same training data.',
                gate_training_exposures_lower_bound=sum(r['exposures'] for r in rows),
                topic_gate_exposures_lower_bound=sum(r['exposures'] for r in rows if r['stage']=='topic-gate-training'),
                relevance_gate_exposures=sum(r['exposures'] for r in rows if r['stage']=='relevance-gate-training'),runs=rows)
    (runs/'registry.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))


if __name__=='__main__':run()
