# Data recipe

The training collection contains 16,288 original human replies, not textbook extracts or
unattributed scraped pages. OpenAssistant provides conversational questions and
follow-ups; Dolly provides many self-contained instruction/answer examples.
Source annotations and length/scope filters are selection aids, not factuality
verification. No user chat is collected or used for training by the page. The public export
omits five answers containing user-specific filesystem path examples, leaving
16,283 displayable records. Those five are training records; held-out query
groups remain unchanged. The export records their IDs and retains all remaining
answer text and attribution.

## Pinned inputs

| Source | Revision | Download SHA-256 |
| --- | --- | --- |
| OpenAssistant/oasst1 | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | `2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7` |
| databricks/databricks-dolly-15k | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec` |

Download `2023-04-12_oasst_all.trees.jsonl.gz` from the pinned OpenAssistant
repository and `databricks-dolly-15k.jsonl` from the pinned Dolly repository.
Verify the hashes above. Keep source licenses with the files.

From the IDI repository root, after installing NumPy, SciPy and scikit-learn:

```sh
export PYTHONPATH=.
python -m idi.layered_chat.openassistant --source /tmp/2023-04-12_oasst_all.trees.jsonl.gz \
  --out idi/layered_chat/data/openassistant-v2
python -m idi.layered_chat.combine_corpora \
  --oa idi/layered_chat/data/openassistant-v2/knowledge.json \
  --dolly /tmp/databricks-dolly-15k.jsonl --out idi/layered_chat/data/combined
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m idi.layered_chat.train_relevance \
  --data idi/layered_chat/data/combined --out /tmp/relevance-run
```

Preparation output directories must be new where the command says they already
exist. The packaged `web/knowledge.json` is also the exact selected collection,
so serving the page does not require downloading upstream data.

OpenAssistant records are English, reviewed, nonsynthetic pairs with qualifying
quality annotations. The best qualifying reply is retained per question. Follow-up
context is limited to 900 characters. Dolly rows are filtered by size, normalized
question duplicates and the declared public-demo topic scope. Individual records
retain their source dataset, revision, ID, license and context.

The paired classifier sees eight lexical quantities quantized to 64 bits. Positive
examples are source questions and deterministic perturbations paired with their
original answers. Other nearby or random answers are weak negatives; some can
also be relevant. TF-IDF hard-negative selection sees the document collection,
including evaluation documents. Query groups, not the document collection, are
held out. Negative documents can come from any split. This is the ordinary
retrieval setting of a fixed searchable collection, not a test of unseen facts.

The combined corpus creates 437,154 training pairs, 53,410 validation pairs and
54,698 test pairs. Balanced minibatches sample negatives to match positives.
Sixty relevance epochs expose 7,494,240 pairs. The held-out pair metric is not a
claim of factual correctness, helpfulness or semantic understanding.

The source collections have broad coverage but do not reliably answer arbitrary
paraphrases. The page requires lexical support, prefers confident learned topic
cards and asks for confirmation when a candidate question contains substantially
more detail than the user's query. A better next dataset would add independently
reviewed paraphrase and ambiguity judgments, grounded multi-turn tasks, explicit
wrong-answer examples and out-of-domain questions. More near-duplicate wrappers
would inflate exposure counts without fixing this limitation.
