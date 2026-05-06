# Political Bias Shareable

Steering LLM political bias via activation engineering, with multilingual transfer evaluation.

## What This Does

1. Train an English `left` or `right` ideology vector from paired corpora using RFM.
2. Apply that vector to steer an LLM at different coefficients.
3. Run Political Compass tests in multiple languages (en, it, fr, es, de).
4. Compare coordinate shifts and item-level answer changes across languages and directions.

## Quick Start

### Single command — full experiment

```bash
./run_local_mps.sh run_experiment.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input-csv data/ideoinst_clean/ideology_840_opinion.csv \
  --device mps \
  --languages en it fr es de \
  --coefs 0.0 0.8 1.5 2.5 4.0 \
  --output-dir outputs/qwen2.5-7b/run_001
```

This runs the full pipeline:
1. **Detection** — extracts left and right concept directions (saved as `directions.pkl`)
2. **Steering** — answers 62 Political Compass questions per language per coefficient per direction
3. **Report** — generates an HTML comparison report with trajectory plots

### Resume after interruption

Re-run the same command. Completed stages are detected and skipped automatically.

### Run on a different model

```bash
./run_local_mps.sh run_experiment.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --input-csv data/ideoinst_clean/ideology_840_opinion.csv \
  --device mps \
  --languages en it \
  --coefs 0.0 1.5 4.0 \
  --output-dir outputs/mistral-7b/run_001
```

## Pipeline Structure

```
run_experiment.py          # Orchestrator — the only entry point you need
pipeline/
├── detect.py              # Stage 1: concept direction extraction (RFM)
├── steer.py               # Stage 2: steering + Political Compass evaluation
├── report.py              # Stage 3: HTML report generation
└── __init__.py
```

### Supporting modules (shared utilities)

```
step1_dataset.py                    # Dataset loading, pairing, topic balancing
political_compass.py                # English compass logic, choice parsing
political_compass_multilingual.py   # Multilingual compass prompts and parsers
prompt_templates.py                 # Prompt template registry
repo_paths.py                      # Locates neural_controllers_official
```

## Output Structure

Each run produces a self-contained directory:

```
outputs/<model>/<run_tag>/
├── config.json              # Full experiment config (reproducible)
├── detection_left/
│   ├── summary.json         # Best layer, AUC metrics
│   ├── directions.pkl       # Reusable concept directions
│   └── detector_coefs.pkl   # Reusable detector coefficients
├── detection_right/
│   └── (same)
├── steering_left/
│   ├── global_summary.json
│   └── <lang>/
│       ├── compass_answers_coef_*.csv
│       └── compass_summary.json
├── steering_right/
│   └── (same)
├── comparison/
│   ├── left_right_report.html
│   └── report_metadata.json   # detection_root == run_root (self-contained)
└── logs/
    └── experiment.log
```

## CLI Reference

```
run_experiment.py
  --model             HuggingFace model name (required)
  --input-csv         Paired ideology CSV (required)
  --output-dir        Where to write results (required)
  --device            auto | mps | cuda | cpu (default: auto)
  --languages         Space-separated language codes (default: en it)
  --coefs             Steering coefficients (default: 0.0 0.8 1.5 2.5 4.0)
  --control-method    rfm | pca | mean_difference (default: rfm)
  --prompt-format     opinion | party (default: opinion)
  --train-pairs       Number of training pairs (default: 100)
  --val-pairs          Number of validation pairs (default: 50)
  --test-pairs         Number of test pairs (default: 50)
  --seed              Random seed (default: 0)
```

## Environment Setup

### Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Apple Silicon (MPS)

Use the wrapper script:

```bash
./run_local_mps.sh run_experiment.py [args...]
```

This locates a Python with `torch`, sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, and runs the script.

RFM fitting runs on CPU; model inference runs on MPS. This is expected.

### External dependency

This repo requires `neural_controllers_official` at one of:
- `<repo-parent>/neural_controllers_official`
- `~/Documents/Playground/neural_controllers_official`

## Multi-Layer Steering

After the main pipeline completes (so `directions.pkl` exists), you can steer
across all layers simultaneously:

```bash
for SIDE in left right; do
  for LANG in en it fr es de; do
    python run_multilayer_steer.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --detection-dir outputs/qwen2.5-7b/run_001/detection_${SIDE} \
      --output-base-dir outputs/qwen2.5-7b/run_001/multilayer_${SIDE} \
      --all-layers \
      --coefs 0.0 0.5 1.0 1.5 2.0 \
      --language $LANG \
      --device auto
  done
done
```

## Models Tested

| Model | HuggingFace ID | GPU Memory |
|-------|---------------|------------|
| Qwen 2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct` | ~16 GB |
| Mistral 7B Instruct v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | ~16 GB |
| Llama 3 8B Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | ~18 GB |

## Data

- `data/ideoinst_clean/ideology_840_opinion.csv` — canonical 840-pair ideology corpus
- `data/political_compass_*_2026.json` — Political Compass questions (en, it, fr, es, de)

## Figure Generation

Visualization scripts (all read from `outputs/` and write to `outputs/figures_*/`):

```
plot_utils.py                       # Shared constants and helpers
plot_main_figures.py                # Main experiment figures
plot_research_story_figures.py      # Research narrative figures
plot_instruct_detection_metrics.py  # Detection AUC visualizations
plot_instruct_stance_coefficients.py # Stance distribution analysis
plot_multilayer_compass.py          # All-layer steering compass plots
```
