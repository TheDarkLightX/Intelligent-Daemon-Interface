# Layered Table Agent

A CPU-trained Boolean-table agent that runs in a browser. Its topic network has
111,104 learned two-input tables, arranged in four layers. The selected variant adds 62 learned integer class offsets. A separate 3,072-table
network scores lexical relevance. The page combines these decisions with 62
original topic cards, 16,283 attributed human answers, and 14 exact algorithm
tools. It shows which route produced each response.

[Open the lab](https://thedarklightx.github.io/Formal_Methods_Philosophy/labs/layered-table-agent/).
[Measurements](web/report.json) identify the deployed checkpoint and evaluation
scope. [Research record](RESULTS.md) lists every attempted training configuration.

The tables select and combine decisions. They do not synthesize unrestricted
new prose. Retrieved answers can be dated, inaccurate or irrelevant. The page
shows their original question, context, dataset and license; loose matches ask
for confirmation. A topic explanation supports follow-ups such as `example`,
`complexity` and `pitfall`. Calculation results include execution steps.

## Run locally

From the repository root:

```sh
python -m http.server 8000 --directory idi/layered_chat/web
```

Open `http://localhost:8000`. All inference and retrieval stay in a Web Worker.
There is no remote inference API. The independent WebAssembly evaluator is
832 bytes; JavaScript provides the same integer semantics as a fallback.
The compressed model is about 290 KB and the answer collection is about 5.9 MB.
Tau is optional. The selected model scores 99.46% on held-out wrappers using
known aliases and 48.39% on unseen descriptions; these are different tests,
not an unrestricted chat-accuracy claim.

## Learn, freeze, fold, replay

The architecture follows the supplied `layered-lookup-tables-v4(1).pdf` and its
training archives: each two-input table relaxes over the 16 Boolean functions;
training fits these choices and freezing selects one function per table.
A truth table is stored as `f00 f01 f10 f11`, most-significant bit first.

The topic wiring learns positive feature associations using training rows only.
Subsequent layers use mostly within-class connections, with a 10% cross-class
sample. Each class receives 448 votes from the fourth layer. The encoder uses
3,072 word, bigram and character-trigram bits. The vote-offset extension is fitted on training rows and rounded to integers. A
common offset makes biases nonnegative without changing margins. The relevance encoder uses 64
integer threshold bits for eight lexical relations.

`graph.py` eliminates constants and wires, folds complements, shares identical
nodes, and removes nodes that cannot affect an output. The trained table count
and executed node count are reported separately. This preserves the frozen
function; it does not retrain a smaller model.

`check_certificate.py` independently checks every local rewrite by enumerating
at most four signal bits, compares exact class-vote multisets, and checks the
dead-node projection. Induction over the original DAG extends these checks to
all Boolean input vectors. This trusts a small Python checker and its execution;
it is not a proof-assistant kernel certificate. The native Tau experiment also
checks the 64 primitive gate applications, 576 folding identities and 16
fresh semantic mutations. Browser parity compares every class vote.

## Data and licenses

| Ingredient | Use | License / provenance |
| --- | --- | --- |
| 62 original, AI-assisted cards | Topic training and displayed explanations | MIT; authored for this experiment |
| 5,211 OpenAssistant pairs | Retrieval and relevance training | Apache 2.0, original notices retained |
| 11,077 Dolly pairs | Retrieval and relevance training | CC BY-SA 3.0 |
| Supplied layered-table paper and archives | Architecture, table order, folding and Tau experiments | User-supplied research; sources named in the record |
| The Algorithm Design Manual | Algorithm selection and independent implementations | No book passages in the corpus or page |

See [NOTICE](web/NOTICE.txt), the per-record source and license fields in
the compressed answer collection, and [pinned source hashes](data/combined/knowledge-report.json).
The combined collection and learned relevance component are CC BY-SA 3.0; code,
authored cards and the topic model are MIT. No source endorsement is implied.

Training contains 62,280 original synthetic topic prompts. Validation and test
wrappers share aliases with training; the 62 unseen descriptions form a separate
challenge. Relevance splits group OpenAssistant conversation trees or Dolly
rows. Its labels distinguish an original paired answer from other documents,
which may also be relevant. The search index includes evaluation documents.
These are retrieval experiments, not a factuality or general-chat benchmark.

## Reproduce

Python 3.12, NumPy, SciPy and scikit-learn are needed for training. Tests also
use pytest. Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` to match the campaign.
Rebuild the data using DATA.md before exporting to a new directory. Run these from the repository root:

```sh
export PYTHONPATH=.
python -m idi.layered_chat.corpus
python -m idi.layered_chat.train --out /tmp/topic-run --votes 448 --depth 4 \
  --epochs 30 --seed 1 --encoder words-first --features 3072 --anneal --rate .025
python -m idi.layered_chat.calibrate_readout --checkpoint /tmp/topic-run/model.json \
  --out /tmp/topic-calibrated
mkdir -p /tmp/browser-assets
cp idi/layered_chat/web/evaluator.wasm /tmp/browser-assets/evaluator.wasm
python -m idi.layered_chat.export --checkpoint /tmp/topic-calibrated/model.json \
  --relevance idi/layered_chat/runs/relevance-source-frozen/model.json \
  --data idi/layered_chat/data/combined --out /tmp/browser-assets --final
python -m pytest idi/layered_chat/tests/test_graph.py
node --test idi/layered_chat/tests/runtime.test.mjs
```

The export directory must already contain `evaluator.wasm` for the asset report;
copy the independent file from `web/` or build it with Clang. Corpus rebuilding
is documented in [DATA.md](DATA.md). `runs/` contains configurations, source
hashes, validation histories and frozen checkpoints. Identical frozen inputs
produce identical integer outputs; floating-point training reproducibility
also depends on numerical library and platform versions.

## Tau and ZenoFCIS

[TAU.md](TAU.md) describes user-loaded Tau, literal graph fragments, the native
strategy-table feature and the boundary between them. No Tau implementation is
included in this repository or page.

ZenoFCIS 2.7.0 source commit
`4160086fb9d19f3ebe25047e69b1caa467aa74fd` supplies ZAL word-kernel synthesis.
All 16 Boolean word kernels were synthesized, emitted and replayed, then used
by a native evaluator to process 64 independent Boolean inputs per word.
The registered ISA model proves the returned u64 word, not a complete ABI,
compiler, browser or speed theorem. The browser uses portable C compiled to
Wasm; x86 kernels are a separate CPU experiment. Native finite-loop containment
was unavailable in this environment, and its failed attempt is retained.

∃AI: Dana Edwards oversees and refines the results.

The canonical answer JSON is stored as `knowledge.json.gz`. Six raw shards provide a fallback for browsers without gzip streams; both paths are checked against the same canonical SHA-256.
