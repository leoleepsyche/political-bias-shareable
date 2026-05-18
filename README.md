# Political Bias Detection & Activation Steering

Detect political ideology directions in LLM hidden states, then steer model outputs via activation engineering. Evaluate with multilingual Political Compass.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (auto-detects GPU/MPS/CPU)
python run.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --input-csv data/ideoinst_clean/ideology_840_opinion.csv \
  --output-dir outputs/mistral_experiment
```

That's it. The script will:
1. Load model, detect device (CUDA/MPS/CPU)
2. Extract hidden states from **all layers**, train ideology probe (RFM)
3. Steer Political Compass answers at coefs [0, 0.8, 1.5, 2.5, 4.0]
4. Save everything to `--output-dir`

## Project Structure

```
run.py                              # Single entry point (run this)
lib/                                # Bundled neural_controllers library
  neural_controllers.py             # NeuralController class
  direction_utils.py                # Hidden state extraction, probes
  generation_utils.py               # Hooked generation for steering
  control_toolkits.py               # RFM/PCA/MeanDiff toolkits
  utils.py                          # Helpers
step1_dataset.py                    # Dataset loading & pairing
political_compass.py                # English compass parsing
political_compass_multilingual.py   # Multilingual compass (en/it/fr/es/de)
data/
  ideoinst_clean/ideology_840_opinion.csv  # 840 paired left/right ideology texts
  political_compass_*.json          # Compass questionnaires (5 languages)
```

## CLI Options

```
python run.py
  --model             HuggingFace model name (required)
  --input-csv         Paired ideology CSV (required)
  --output-dir        Output directory (required)
  --languages         Language codes (default: en it)
  --coefs             Steering coefficients (default: 0.0 0.8 1.5 2.5 4.0)
  --control-method    rfm | pca | mean_difference (default: rfm)
  --train-pairs       Training pairs per side (default: 100)
  --val-pairs         Validation pairs (default: 50)
  --test-pairs        Test pairs (default: 50)
  --n-components      Probe components (default: 1)
  --rfm-iters         RFM iterations (default: 8)
  --batch-size        Forward pass batch size (default: 1)
  --seed              Random seed (default: 0)
```

## Output Structure

```
outputs/my_experiment/
├── config.json
├── run.log
├── detection_left/
│   ├── summary.json          # Best layer, AUC
│   ├── directions.pkl        # ALL layer directions
│   └── detector_coefs.pkl    # ALL layer detector coefficients
├── detection_right/
│   └── (same)
├── steering_left/              # Best-layer steering
│   ├── global_summary.json
│   └── en/
│       ├── compass_answers_coef_0.csv
│       ├── compass_answers_coef_0.8.csv
│       └── compass_summary.json
├── steering_right/             # Best-layer steering
│   └── (same)
├── steering_left_alllayers/    # All-layers steering
│   └── (same structure)
└── steering_right_alllayers/   # All-layers steering
    └── (same structure)
```

The pipeline automatically runs both steering modes:
- **Best-layer**: inject direction at the single best detection layer
- **All-layers**: inject direction at every layer simultaneously

## Device Compatibility

| Device | Status | Notes |
|--------|--------|-------|
| CUDA   | Full speed | RFM runs on GPU |
| MPS (Apple Silicon) | Supported | RFM runs on CPU, inference on MPS |
| CPU    | Supported | Slow but works |

The script auto-detects your device and patches the library accordingly.

## How It Works

1. **Dataset**: Paired left/right ideology responses (from IdeoINST corpus)
2. **Detection**: Feed pairs through LLM → extract hidden states at all layers → train RFM probe per layer → find "ideology direction"
3. **Steering**: At generation time, add `coef * direction` to the hidden state at the best layer → answer Political Compass
4. **Evaluation**: Parse choices, compute economic (x) and social (y) coordinates

## Models Tested

- `mistralai/Mistral-7B-Instruct-v0.3`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Meta-Llama-3-8B-Instruct`

## Dependencies

- `torch >= 2.0`
- `transformers >= 4.30`
- RFM/xRFM is bundled in `lib/xrfm` (do not install PyPI `xrfm`)
- `scikit-learn >= 1.0`
- `tqdm`
- `numpy`

## Resumable

Re-run the same command if interrupted — completed stages are auto-skipped.
