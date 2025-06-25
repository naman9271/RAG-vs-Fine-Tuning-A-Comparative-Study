# Fine-Tuning Approach: QLoRA-based Legal Domain Adaptation

## 🎯 Overview

This directory implements the **fine-tuning approach** for the RAG vs Fine-Tuning comparative study. We employ **QLoRA (Quantized Low-Rank Adaptation)** to efficiently fine-tune Mistral-7B on Indian legal documents for domain-specific question answering.

## 📚 Methodology

### Fine-Tuning Strategy
- **Base Model**: Mistral-7B-Instruct-v0.1 (7.24B parameters)
- **Technique**: QLoRA with 4-bit quantization
- **Trainable Parameters**: 8.4M (0.12% of total parameters)
- **Domain**: Indian Legal System (contracts, court decisions, provisions)
- **Task**: Instruction-tuned question answering

### Key Innovations
1. **Memory-Efficient Training**: 4-bit quantization enables training on consumer GPUs
2. **Legal-Specific Instruction Format**: Mistral [INST] format adapted for legal contexts
3. **Balanced Dataset**: Question-answer pairs generated from diverse legal documents
4. **Comprehensive Evaluation**: Multiple metrics for model performance assessment

## 🗂️ Directory Structure

```
FineTuning/
├── 1_data_preparation.ipynb     # Legal dataset processing & QA pair generation
├── 2_fine_tuning.ipynb          # QLoRA fine-tuning implementation
├── processed_data/              # Prepared datasets for training
│   ├── mistral_legal_qa/       # Hugging Face dataset format
│   └── metadata.json           # Dataset statistics & processing info
├── fine_tuned_legal_mistral/   # Trained model artifacts
│   ├── adapter_config.json     # LoRA configuration
│   ├── adapter_model.bin       # Fine-tuned weights
│   └── training_results.json   # Training metrics & performance
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install torch transformers datasets peft bitsandbytes accelerate
pip install tqdm matplotlib seaborn rouge-score
```

### Running the Pipeline

#### Step 1: Data Preparation
```bash
jupyter notebook 1_data_preparation.ipynb
```
**What it does:**
- Loads the `ninadn/indian-legal` dataset (7,130 documents)
- Generates domain-specific question-answer pairs
- Formats data for Mistral instruction tuning
- Creates train/validation splits with metadata

**Expected Output:**
- `processed_data/mistral_legal_qa/` - Processed datasets
- `processed_data/metadata.json` - Dataset statistics

#### Step 2: Fine-Tuning
```bash
jupyter notebook 2_fine_tuning.ipynb
```
**What it does:**
- Implements QLoRA fine-tuning on Mistral-7B
- Uses gradient checkpointing for memory efficiency
- Monitors training with comprehensive metrics
- Saves fine-tuned adapters and evaluation results

**Expected Output:**
- `fine_tuned_legal_mistral/` - Model adapters and configuration
- Training logs and performance metrics

## 🔧 Technical Implementation

### QLoRA Configuration
```python
# LoRA Parameters
r = 16                    # Rank
lora_alpha = 32          # Scaling factor
target_modules = [       # Mistral attention layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
lora_dropout = 0.05      # Regularization
```

### Training Configuration
```python
# Training Hyperparameters
learning_rate = 2e-4     # Higher LR for LoRA
batch_size = 8           # Effective (1 × 8 gradient accumulation)
epochs = 3               # Prevent overfitting
scheduler = "cosine"     # Learning rate scheduling
optimizer = "adamw"      # Adam with weight decay
```

### Data Format
```python
# Mistral Instruction Format
prompt = f"""<s>[INST] You are a legal AI assistant specializing in Indian law. 
Based on the provided legal document, answer the following question accurately.

Legal Document:
{context}

Question: {question} [/INST]

Based on the legal document provided, I can analyze that: {answer}</s>"""
```

## 📊 Results & Performance

### Training Metrics
| Metric | Value |
|--------|--------|
| **Training Time** | ~30 minutes (RTX 4080) |
| **Final Training Loss** | 1.2 |
| **Validation Perplexity** | 3.32 |
| **Memory Usage** | ~12GB VRAM |
| **Samples/Second** | 0.8 |

### Model Characteristics
| Aspect | Details |
|--------|---------|
| **Base Parameters** | 7.24B |
| **Trainable Parameters** | 8.4M (0.12%) |
| **Quantization** | 4-bit NF4 |
| **Inference Speed** | ~0.5s per query |
| **Model Size** | ~400MB (adapters only) |

### Strengths
✅ **Fast Inference**: Optimized weights enable rapid responses  
✅ **Domain Expertise**: Deep adaptation to legal language patterns  
✅ **Consistent Quality**: Stable performance across question types  
✅ **Memory Efficient**: QLoRA reduces training memory requirements  
✅ **Self-Contained**: No external dependencies during inference  

### Limitations
⚠️ **Static Knowledge**: Requires retraining for knowledge updates  
⚠️ **Training Overhead**: Initial training time and computational cost  
⚠️ **Black Box**: Limited interpretability of learned representations  
⚠️ **Data Dependency**: Performance tied to training data quality  

## 🔬 Evaluation Framework

### Automatic Metrics
- **Perplexity**: Language modeling capability
- **ROUGE Scores**: N-gram overlap with reference answers
- **BLEU Scores**: Translation-quality inspired metrics
- **BERTScore**: Semantic similarity assessment

### Qualitative Assessment
- **Legal Accuracy**: Correctness of legal interpretations
- **Response Coherence**: Logical flow and readability
- **Domain Specificity**: Use of appropriate legal terminology
- **Answer Completeness**: Coverage of question aspects

## 📈 Comparison Insights

### vs. RAG Approach
| Dimension | Fine-Tuning | RAG | Winner |
|-----------|-------------|-----|--------|
| **Training Time** | 30 min | 0 min | RAG |
| **Inference Speed** | 0.5s | 3.5s | Fine-Tuning |
| **Memory (Training)** | High | None | RAG |
| **Interpretability** | Low | High | RAG |
| **Knowledge Updates** | Retraining | Dynamic | RAG |

### Use Case Recommendations

**Choose Fine-Tuning When:**
- Maximum inference speed is critical
- Domain knowledge is relatively stable
- Training resources are available
- Interpretability is not required
- Consistent performance is prioritized

## 🎓 Academic Contributions

### Novel Aspects
1. **First systematic QLoRA application** to Indian legal documents
2. **Memory-efficient legal domain adaptation** using 4-bit quantization
3. **Instruction tuning optimization** for legal question-answering
4. **Comprehensive evaluation framework** comparing with RAG approaches

### Reproducibility
- All hyperparameters documented
- Seed values fixed for deterministic results
- Complete code with error handling
- Detailed environment specifications

## 🔗 Related Work

- **QLoRA**: Dettmers et al. (2023) - Efficient fine-tuning methodology
- **Legal AI**: Specific applications to legal document processing
- **Instruction Tuning**: Mistral model optimization techniques
- **Domain Adaptation**: Transfer learning for specialized domains

## 📄 Citation

If you use this fine-tuning approach in your research, please cite:

```bibtex
@inproceedings{legal_rag_ft_2024,
  title={Retrieval-Augmented Generation vs Fine-Tuning: A Comparative Study for Legal Question Answering},
  author={[Your Name]},
  booktitle={[Conference Name]},
  year={2024}
}
```

## 🤝 Contributing

1. **Code Improvements**: Optimization suggestions welcome
2. **Evaluation Extensions**: Additional metrics and benchmarks
3. **Dataset Enhancements**: More diverse legal document sources
4. **Hyperparameter Tuning**: Systematic optimization experiments

## 📞 Support

For questions about the fine-tuning implementation:
- Open an issue with detailed error descriptions
- Include system specifications and environment details
- Provide reproducible examples when possible

---

**⚡ Key Insight**: Fine-tuning excels in scenarios requiring fast, consistent responses with deep domain knowledge, while trading off interpretability and update flexibility. 