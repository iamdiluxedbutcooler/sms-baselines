# SMS Baselines

Comparative study of SMS classification methods for spam/ham detection. Implements baselines from academic papers using identical train/test splits.

## Structure

```
sms-baselines/
├── configs/          # YAML configuration files
├── data/splits/      # Train/test CSV files
├── experiments/      # Results and trained models
├── notebooks/        # Jupyter notebooks for execution
└── src/models/       # Baseline implementations
    ├── nlp/         # Traditional ML methods
    ├── nn/          # Neural network approaches
    └── llm/         # Large language models
```

## Baselines

### Traditional ML
- **bl_nlp_01**: TF-IDF + Multinomial Naive Bayes (Mishra 2020)
- **bl_nlp_02**: RoBERTa with preprocessing (Salman 2024)

### Neural Networks
- **bl_nn_01**: CNN-BiGRU with Word2Vec (Mahmud 2024)
- **bl_nn_02**: Character-level CNN with masking (Seo 2024)

### Large Language Models
- **bl_llm_01**: Multi-model comparison (GPT-4o, Llama-3, Mistral)
- **bl_llm_02**: Fine-tuned Mixtral-8x7B with QLoRA (Salman 2025)

## Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn torch transformers peft bitsandbytes openai
```

### Running Experiments

1. **Setup**: Mount Google Drive and clone repository
2. **Execute**: Run notebook 02 to train all baselines
3. **Analyze**: Run notebook 03 to generate results and visualizations

### Configuration

Each baseline has a YAML config file specifying:
- Paper details and methodology
- Model hyperparameters
- Training settings
- Required libraries

### Results

Each run produces:
- Accuracy metrics
- Classification reports
- Confusion matrices
- Detailed predictions CSV
- Trained model files

## Features

- **Reproducible**: Fixed train/test splits and seeded runs
- **Scalable**: Modular design for adding new baselines
- **Robust**: Fallback mechanisms for failed training
- **Automated**: Git integration and checkpoint saving
- **Comprehensive**: Multiple runs per baseline for variance estimation

## Hardware Requirements

- **Traditional ML**: CPU sufficient
- **Neural Networks**: GPU recommended (T4/L4/A100)
- **Large Language Models**: High-memory GPU required (A100/L4)

## Adding New Baselines

1. Create implementation file in appropriate `src/models/` subdirectory
2. Add YAML configuration in `configs/`
3. Follow existing naming convention: `bl_[type]_[number]`
4. Implement `run_bl_[type]_[number]()` function with standard signature

## Data Format

Training and test files must contain:
- `text`: SMS message content
- `label`: Classification (spam/ham)

## Authentication

- **GitHub**: Personal access token prompted during execution
- **OpenAI**: API key prompted when using GPT-4o baseline

## Output

Results saved to `experiments/` directory:
- Individual run folders with detailed results
- Baseline summary CSV files
- Aggregated analysis across all baselines
- Performance visualizations and error analysis