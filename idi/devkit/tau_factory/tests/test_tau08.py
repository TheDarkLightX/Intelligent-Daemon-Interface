from idi.devkit.tau_factory.tau08 import migrate,validate_file_spec,without_comment
from idi.devkit.tau_factory.generator import generate_tau_spec,generate_legacy_tau_spec
from idi.devkit.tau_factory.schema import AgentSchema,StreamConfig,LogicBlock

def test_hex_constants_and_quoted_hash_are_not_comments():
    assert without_comment('x = {#xFF}:bv[8]. # note')=='x = {#xFF}:bv[8]. '
    assert without_comment('i:sbf := in file("a#b.in"). # note')=='i:sbf := in file("a#b.in"). '
    m=migrate('i:bv[8] = in file("a.in")\no:bv[8] = out file("b.out")\nr o[t] = {#xFF}:bv[8].\nq\n')
    assert '{#xFF}:bv[8]' in m.text
    assert validate_file_spec(m.text)==(True,[])

def test_default_is_file_dialect_and_legacy_is_explicit():
    s=AgentSchema(name='copy',strategy='copy',streams=(StreamConfig('data','sbf'),StreamConfig('copy','sbf',is_input=False)),logic_blocks=(LogicBlock('passthrough',('data',),'copy'),),include_mirrors=False)
    modern=generate_tau_spec(s)
    assert ':= in file(' in modern and 'always (' in modern and '\nq' not in modern
    assert generate_tau_spec(s,dialect='0.7')==generate_legacy_tau_spec(s)
    assert validate_file_spec(modern)==(True,[])

def test_repeated_runs_are_refused_for_manual_review():
    assert 'multiple run commands' in migrate('r o[t]=0.\nr o[t]=1.').issues

def test_recorded_reference_schemas_regenerate_exactly():
    import json
    from pathlib import Path
    root=Path(__file__).parent/'tau08_cases'
    for row in json.loads((root/'cases.json').read_text()):
        s=dict(row['schema']);s['streams']=tuple(StreamConfig(**v) for v in s['streams']);s['logic_blocks']=tuple(LogicBlock(**dict(v,inputs=tuple(v['inputs']))) for v in s['logic_blocks'])
        assert generate_tau_spec(AgentSchema(**s))==(root/(row['pattern']+'.tau')).read_text(),row['pattern']
