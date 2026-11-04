import itertools
import numpy as np
from idi.layered_chat.graph import apply,evaluate,fold
from idi.layered_chat.network import gradient_check


def test_all_table_entries_against_boolean_minterms():
    for f,a,b in itertools.product(range(16),range(2),range(2)):
        expected=any(((f>>(3-2*x-y))&1) and a==x and b==y for x,y in itertools.product(range(2),repeat=2))
        assert apply(f,a,b)==expected


def test_folding_exhaustive_random_dags():
    # Every assignment for 128 independently sampled 8-input graphs.
    x=np.array(list(itertools.product((0,1),repeat=8)),np.uint8)
    for seed in range(128):
        rng=np.random.default_rng(seed)
        nodes=[[int(rng.integers(0,8+j)),int(rng.integers(0,8+j)),int(rng.integers(16))] for j in range(64)]
        votes=[[[int(rng.integers(72)),int(rng.integers(2))] for _ in range(16)] for c in range(3)]
        g=dict(inputs=8,classes=3,nodes=nodes,votes=votes,bias=[1,2,3])
        assert np.array_equal(evaluate(g,x),evaluate(fold(g),x))


def test_folding_catches_changed_live_gate():
    g=dict(inputs=2,classes=1,nodes=[[0,1,6]],votes=[[[2,0]]],bias=[0])
    bad=dict(g,nodes=[[0,1,7]])
    x=np.array(list(itertools.product((0,1),repeat=2)))
    assert not np.array_equal(evaluate(g,x),evaluate(bad,x))


def test_gradient_independent_central_difference():
    assert gradient_check('choice',2)<1e-7

def test_independent_certificate_replay_and_semantic_mutations():
    from idi.layered_chat.check_certificate import replay,Refused
    import copy,pytest
    for seed in range(40):
        rng=np.random.default_rng(seed+400)
        g=dict(inputs=5,classes=2,nodes=[[int(rng.integers(5+i)),int(rng.integers(5+i)),int(rng.integers(16))] for i in range(40)],votes=[[[43,0],[44,1]],[[42,1],[40,0]]],bias=[0,2])
        certificate={};f=fold(g,certificate);assert replay(g,f,certificate)['classes']==2
        bad=copy.deepcopy(certificate);bad['bias'][0]+=1
        with pytest.raises(Refused):replay(g,f,bad)

def test_ensemble_composition_preserves_every_integer_vote():
    from idi.layered_chat.combine_models import combine
    from idi.layered_chat.train import scores
    rng=np.random.default_rng(145);models=[]
    for votes in (3,5,7):
        widths=[4,2*votes,2*votes];layers=[]
        for before,after in zip(widths,widths[1:]):layers.append(dict(a=rng.integers(before,size=after).tolist(),b=rng.integers(before,size=after).tolist(),f=rng.integers(16,size=after).tolist()))
        models.append(dict(schema='idi/layered-chat/1',table_order='msb-00-01-10-11',inputs=4,features=list('abcd'),classes=2,ids=['x','y'],layers=layers))
    x=np.array(list(itertools.product(range(2),repeat=4)),np.uint8)
    assert np.array_equal(scores(combine(models),x),sum(scores(m,x) for m in models))

def test_byte_features_preserve_float32_wiring_loss_and_gradients():
    from idi.layered_chat.network import Network
    from idi.layered_chat.train import wire_guided
    rng=np.random.default_rng(20261104);x=rng.integers(0,2,(96,80),dtype=np.uint8);y=np.arange(96)%3
    a,b=Network(80,[48]*4,'choice',2,71),Network(80,[48]*4,'choice',2,71)
    wire_guided(a,x,y,3,16);wire_guided(b,x.astype(np.float32),y,3,16)
    for la,lb in zip(a.layers,b.layers):assert np.array_equal(la.wires,lb.wires)
    for step in range(5):
        loss_a,ga=a.loss_and_grads(x,y,3,1.6);loss_b,gb=b.loss_and_grads(x.astype(np.float32),y,3,1.6)
        assert loss_a==loss_b
        for la,lb,g1,g2 in zip(a.layers,b.layers,ga,gb):
            assert np.array_equal(g1,g2);la.adam(g1,.025);lb.adam(g2,.025);assert np.array_equal(la.logits,lb.logits)
    assert x.nbytes*4==x.astype(np.float32).nbytes

def test_integer_readout_offsets_survive_folding_and_certificate_replay():
    from idi.layered_chat.train import scores
    from idi.layered_chat.graph import from_layers, fold, evaluate
    from idi.layered_chat.check_certificate import replay
    model = {'inputs':3, 'classes':2, 'layers':[{'a':[0,0,1,2], 'b':[1,2,2,0], 'f':[1,3,7,9]}], 'readout_bias':[0,7]}
    x = np.array([[i >> j & 1 for j in range(3)] for i in range(8)], dtype=np.uint8)
    original = from_layers(model)
    certificate = {}
    folded = fold(original, certificate)
    assert np.array_equal(scores(model, x), evaluate(folded, x))
    assert replay(original, folded, certificate)['classes'] == 2
