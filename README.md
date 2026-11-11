# JIP: Judge-Imitating Probe

Code for the paper *Imitate the Judge, Keep the Probe: Usable Information for Uncertainty Quantification in LLM
Reasoning*.

JIP estimates whether an LLM's chain-of-thought answer is correct. A small probe reads the frozen hidden states of
the generating model and outputs a correctness score. During training it also imitates an answer-blind LLM judge:
it predicts the judge's verdict level and the judge's label of each reasoning step. These imitation outputs need no
correctness label, so the probe also learns from unlabeled questions of other domains, and they sit beside the
correctness output instead of replacing it. A calibrator combines the probe's outputs with a second probe (SAPLMA)
and features of the reasoning chain. **JIP** uses no external model at inference; **JIP + Judge** adds one judge
call per response, and calls can be routed to the responses with the largest expected information gain.

## Installation

Python 3.10 or newer and a CUDA build of PyTorch are required. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

The package is imported in place: run every command from the repository root (`python -m jip.<module>`).

## Configuration

`.env` holds all paths, devices and credentials; see [`.env.example`](.env.example). Variables exported in the
shell take precedence, and relative paths resolve against the repository root.

```dotenv
JIP_ARTIFACTS=artifacts                          # generations, judge outputs, features, probes, results
JIP_POOL_ARTIFACTS=${JIP_ARTIFACTS}/pools        # 6,000-question unlabeled pools
JIP_DATASETS=datasets
JIP_MODELS=models
JIP_GEN_GPU=0
JIP_JUDGE_GPU=1
```

## Data and models

`python -m jip.pipeline.prepare_data --pools 2000,6000` downloads the benchmarks (HotpotQA, MuSiQue, StrategyQA,
bAbI, GSM8K, MATH) from the Hugging Face Hub and freezes every split under `JIP_DATASETS`.

Place the models under `JIP_MODELS`, one directory per model, named as in [`jip/registry.py`](jip/registry.py):
Llama-3.1-8B-Instruct, Qwen3-8B and Ministral-8B-Instruct-2410 (backbones), gpt-oss-20b (judge), and optionally
gpt-oss-120b and Qwen3-4B-Instruct-2507 (further judges). The Claude Haiku judge
(`jip.pipeline.judge_api`) needs `ANTHROPIC_API_KEY` in `.env` or in the environment.

## Running the pipeline

One backbone, all datasets (each stage calls the `jip.pipeline` modules; see the script for the commands):

```bash
python -m jip.pipeline.prepare_data --pools 2000,6000
scripts/serve_model.sh gpt-oss-20b &       # judge server on JIP_JUDGE_GPU
scripts/run_backbone.sh llama gen          # greedy generation and feature extraction
scripts/run_backbone.sh llama judge        # correctness labels, step labels and verdicts of the judge
scripts/run_backbone.sh llama pools        # unlabeled pools of the other domains: generation, judge targets, features
scripts/run_backbone.sh llama probes       # SAPLMA and JIP's probe, 10 seeds each
python -m jip.pipeline.evaluate --backbone llama
```

`jip.pipeline.train` names JIP's probe `jip`; `jip-source` is the same probe
with imitation on the labeled source only. `jip.pipeline.evaluate` fits the calibrator on the validation split of
each dataset and writes the results of JIP and JIP + Judge to `results/<backbone>/eval.json` under `JIP_ARTIFACTS`
(`--split test2` for the second held-out split).

All three backbones, end to end (several GPU-days on two 80 GB GPUs):

```bash
scripts/run_main.sh
```

## Repository layout

```text
jip/
  config.py            dotenv loader and paths
  registry.py          backbones, judges, datasets, run and result keys
  artifacts.py         artifact layout and JSONL I/O
  data/                dataset loaders, prompts, answer matching, response parsing
  probes/              JIP's probe (token and layer branches, imitation outputs) and SAPLMA
  training.py          probe training (correctness and imitation losses) and prediction
  samples.py           per-response records from generations, judge outputs and features
  chain_features.py    chain features u(r) and span token features
  calibrator.py        JIP's calibrator and the scalar calibrators it builds on
  evaluation.py        evidence assembly and scoring of one evaluation cell
  routing.py           routing of judge calls by expected information gain
  metrics.py           discrimination and calibration metrics
  pipeline/            prepare_data, generate, judge, judge_api, features, train, evaluate, audit
scripts/               driver scripts (configured by .env)
```

## License

Released under the [MIT License](LICENSE). Models and datasets keep their own licenses.
