# CPU layered-table conversation experiment

The released agent contains **111,104 trained topic tables plus 3,072 relevance tables**, with 62 learned integer vote offsets. Exact folding leaves **39,603 topic gates and 1,194 relevance gates**. The compressed model is **289,606 bytes** and the compressed answer collection is **5,877,308 bytes**. The evaluator is 832 bytes of independent WebAssembly.

The selected checkpoint is `calibrated-111104-annealed`, SHA-256 `f2ba5e05db45d883aed549f07842d3fb7578cd237842ca78133d8ceb20d58e88`. Selection was frozen before opening test/challenge scores.

| Evaluation | Result | Scope |
| --- | ---: | --- |
| Validation | 656/744 (88.17%) | 744 prompts; used repeatedly for checkpoint/architecture selection |
| Held-out wrappers | 740/744 (99.46%) | 744 prompts; aliases also occur in training |
| Unseen descriptions | 30/62 (48.39%) | 62 new descriptions, one per class |

At the validation-selected margin of 74, the wrapper test accepts 697/744 inputs, all correct in that sample. On the unseen-description challenge it accepts 20/62, of which only 12 are correct. The cutoff does **not** carry its apparent precision to a changed query distribution. The page adds lexical support and abstention, but it does not guarantee relevance.

The corpus contains 62,280 distinct training prompts. The campaign records at least **89,020,920 gate-training presentations**: 62,621,160 topic presentations and 26,399,760 relevance-pair presentations. These include repeated epochs and repeated relevance runs, not that many unique conversations. [Every run](runs/registry.json), [selection](runs/selection.json), and [release measurements](web/report.json) are retained.

Three independent relevance runs reproduced the same frozen model bytes in the same numerical environment; see [reproduction record](runs/relevance-reproduction.json).

## Retrieval and relevance results

The frozen relevance network reaches **99.13% balanced accuracy** on 54,698 weakly labeled held-out pairs. Positive recall is 99.97% and negative recall is 98.29%. Fixed baselines reach 89.62% (query coverage at least one half), 85.28% (both coverage measures at least one half), and 61.34% (exact normalized question). These measure the weak pairing task, not answer correctness.

The separate [end-to-end retrieval report](evidence/retrieval-final/report.json) uses 1,634 source-grouped test questions whose answer records are already in the collection:

| Query condition | BM25 top-1 exact source | Agent directly returns exact source | Other outcomes |
| --- | ---: | ---: | --- |
| Original question | 1,634/1,634 | 1,474/1,634 | 160 topic-card routes |
| Every fifth lexical term removed | 1,630/1,634 | 1,542/1,634 | Four different records, 87 topic cards, one empty-input route |

Both conditions retrieve the exact source within the top 12. A different record can still answer the question; these are exact-ID measurements, not semantic relevance judgments. The four differing IDs and queries are retained. Original-question recovery is an easy transductive retrieval check, not unseen-question generalization.

## Question and design

Can the layered-table design in Dana Edwards's *How Far Can Layered Lookup Tables Go?*, version 4, support a useful browser conversation at more than 100,000 trained tables on CPU?

Each table learns a choice among the 16 two-input Boolean functions. Freezing chooses one function. Class-group outputs become integer votes. A fixed 3,072-bit lexical encoder represents the message. The guided-wiring experiment learns feature associations on training data before fitting table functions. Later layers mostly connect within their class group, with a 10% cross-class sample.

The agent combines this classifier with 62 original algorithm/formal-methods cards, 14 bounded algorithm tools, and an attributed answer collection. Training uses 16,288 OpenAssistant/Dolly records; the page displays 16,283 after excluding five records with user-specific filesystem path examples. A separately trained 3,072-table network judges lexical question/answer-pair relevance. It does not validate the truth of a retrieved answer.

The inputs and their hashes are in [research-inputs.json](research-inputs.json). No textbook passages or Tau implementation are included.

## What the comparisons establish

The matched 15,872-table control reached 91.67% validation accuracy with guided wiring and 38.04% with random wiring. Both used 30 epochs, seed 1, 3,072 words-first features, annealing and learning rate 0.025. This is evidence that the wiring prior matters on these synthetic topic prompts.

A single 111,104-table, four-layer model reached 88.04% before readout calibration. Seven independently trained 15,872-table networks, composed by equal integer votes into the same 111,104-table budget, reached 86.69%. More tables and equal-vote composition did not improve this benchmark.

A training-only extension fits one integer offset per class. These 62 offsets correct score imbalance while preserving every Boolean table. They improve the large annealed model to 88.17% validation accuracy and the ensemble to 87.23%. They are additional learned readout parameters, not extra tables. The optimization uses training rows only; its temperature and regularization are fixed in `calibrate_readout.py`.

A two-layer 111,104-table comparison was stopped after five of 15 planned epochs to reserve publication time. It reached 80.38% before offsets and 80.78% after them. This incomplete run does not establish a general depth advantage. The 60-epoch, four-layer seed-0 run finished; its best raw validation result was 87.77%, and its offsets reduced that to 87.63%.

The earlier 1,536-feature frequency-encoder runs, the initial smaller-data pilot, and the later 3,072-feature runs are different settings. They are not a controlled table-count scaling curve. Validation selected checkpoints and architectures repeatedly, so validation results are optimistic. Held-out wrapper prompts also reuse training aliases; unseen-description results are reported separately.

## Execution and checking

The independent 832-byte WebAssembly evaluator runs the frozen graph with integer semantics. JavaScript is its fallback. Exact folding removes constant, wire, complement, shared and unused nodes. The reported trained-table count and executed-gate count are different quantities.

The independent Python certificate checker enumerates each local substitution over at most four signal bits, verifies exact vote multisets and biases, and verifies the dead-node projection. Induction over the DAG extends this to every Boolean input assignment. This is a checked finite-rewrite argument with an explicit Python checker in the trusted base, not a proof-assistant-kernel claim. Five fresh mutations exercise gate, bias, vote, projection and input-map checks.

A direct whole-class SMT approach timed out on all 62 class-sum queries; those outcomes remain `unknown` in the retained preview evidence. The successful local-induction certificate is a separate route.

Native Tau 0.8 checks all 64 gate applications, 576 folding identities and 16 semantic mutations. Original Tau graph fragments carry explicit external-signal wiring and integer vote maps. Native fragment sampling checks actual node values. The final export contains 1,238 Tau fragments. Native sampling executes 32 of them on two inputs each, checking 2,022 emitted node values. The [browser report](evidence/browser-tau-final/report.json) records an actual separately built Tau Wasm runtime: all 3,596 class scores across 58 inputs agree, as do five conversation paths. The runtime is not redistributed. The [all-input folding certificate](evidence/final-fold/report.json) and [fragment sampling report](evidence/tau-fragments-final/qualification.json) have separate scopes.

ZenoFCIS 2.7.0, pinned at `4160086fb9d19f3ebe25047e69b1caa467aa74fd`, synthesizes and replays all 16 u64 Boolean kernels through ZAL. Native C uses those emitted x86 kernels for 64-lane bit-parallel evaluation and compares its full class scores with Python and portable C. These are returned-word proofs under the registered ISA model plus host differential tests, not an ABI, compiler or browser proof. The final benchmark checked 124,000 class votes over 1,000 inputs across both native evaluators. For batches of 64 inputs, portable C took a median 1.46 ms and the ZAL-kernel host took 1.85 ms, including Python/FFI packing. The ZAL version was slower in this measurement. Other qualification jobs ran concurrently, without process isolation; no browser speed comparison is implied. See the [native report](evidence/native-final/report.json).

## Failures and remaining limits

The environment provided an eight-CPU quota and 8 GiB of memory, with no GPU. A native Tau source build exhausted memory; the pinned official Python wheel supplied native qualification. Four ensemble starts were killed before their first metric. Their attempts are retained. Binary training features were reduced from float32 to uint8 storage; a seeded test checks identical wiring, losses, gradients and optimizer updates for that change. The original exploratory campaign was interrupted to free resources. No credit is given for epochs that did not produce a completed log.

The classifier has lexical features and a small topic ontology. It cannot compose unrestricted new prose. Retrieval is transductive: the collection contains the evaluation documents. Positive/negative pair labels are weak, and near-duplicate questions can be valid alternatives. Pair accuracy and exact document-ID recovery do not measure truthfulness, reasoning or conversational helpfulness.

Richer semantic coverage needs reviewed paraphrases, ambiguity labels, independently checked answers, grounded multi-turn tasks and hard counterexamples. GPU training would shorten some fitting runs; it would not supply those missing judgments.

The factory now emits Tau 0.8 by default. All 122 reference schemas are admitted by the native interpreter; 24 native execution tests cover the factory and IAN contracts. Among 106 historical Tau files, 58 received envelope/type ports, nine were replaced with explicit bounded contracts, and 39 remain historical or inconclusive. The IAN bridge gains pure preflight checks and requires an injected verifier when governance approval is configured. This does not implement transaction finality or the full cryptographic pipeline. See the [migration inventory](../../specification/tau08/README.md) and [regression record](../../specification/tau08/regression-report.json).

The broad IAN suite still has 10 failures and seven setup errors, all reproduced on the original revision. The focused bridge tests and the full factory suite pass in their stated environments. These existing failures are not presented as a green full-repository result.

∃AI: Dana Edwards oversees and refines the results.
