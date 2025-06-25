"""
Configuration Management for RAG vs Fine-Tuning Comparative Study
Academic Research Project Configuration

This module centralizes all configuration parameters for reproducible experiments.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch
import json
from pathlib import Path

@dataclass
class DatasetConfig:
    """Dataset configuration parameters"""
    dataset_name: str = "ninadn/indian-legal"
    sample_size: int = 1000  # For development, increase for full experiments
    train_test_split: float = 0.85
    random_seed: int = 42
    max_document_length: int = 2048
    min_document_length: int = 100
    
    # Data processing
    clean_text: bool = True
    remove_special_chars: bool = True
    normalize_whitespace: bool = True

@dataclass
class FineTuningConfig:
    """Fine-tuning approach configuration"""
    # Model settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    output_dir: str = "./FineTuning/fine_tuned_legal_mistral"
    
    # QLoRA settings
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training parameters
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    
    # Training optimization
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False
    group_by_length: bool = True
    
    # Evaluation
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

@dataclass
class RAGConfig:
    """RAG approach configuration"""
    # Model settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector database settings
    vector_db_path: str = "./RAG/vector_db"
    faiss_index_name: str = "faiss_legal_db"
    chroma_collection_name: str = "indian_legal_documents"
    
    # Document processing
    chunk_size: int = 800
    chunk_overlap: int = 100
    chunk_separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])
    
    # Retrieval settings
    k_retrieve: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 1500
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

@dataclass
class EvaluationConfig:
    """Evaluation configuration for academic comparison"""
    # Metrics
    compute_rouge: bool = True
    compute_bleu: bool = True
    compute_bertscore: bool = True
    compute_perplexity: bool = True
    
    # Statistical testing
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    confidence_interval: float = 0.95
    
    # Evaluation datasets
    test_questions: List[str] = field(default_factory=lambda: [
        "What are the legal obligations of contractors in equipment agreements?",
        "How does the Bihar Sales Tax Act apply to machinery sales?",
        "What is the court's decision regarding contract disputes?",
        "What are the payment terms for equipment leasing agreements?",
        "What legal provisions govern the ownership of machinery?",
        "How are legal disputes between corporations and contractors resolved?",
        "What are the consequences of breaching equipment lease agreements?",
        "What role do consulting engineers play in legal agreements?",
        "What are the remedies available for breach of contract?",
        "How are damages calculated in contractual disputes?"
    ])
    
    # Performance benchmarks
    max_acceptable_latency: float = 10.0  # seconds
    min_acceptable_accuracy: float = 0.7
    target_memory_usage: float = 16.0  # GB

@dataclass
class ExperimentConfig:
    """Experiment tracking and reproducibility"""
    experiment_name: str = "rag_vs_finetuning_legal_qa"
    run_id: Optional[str] = None
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    benchmark: bool = True
    
    # Logging
    log_level: str = "INFO"
    save_logs: bool = True
    log_dir: str = "./logs"
    
    # Results
    results_dir: str = "./results"
    save_models: bool = True
    save_intermediate_results: bool = True
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    compile_model: bool = False

@dataclass
class ResearchConfig:
    """Main configuration class combining all settings"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def save_config(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        config_dict = {
            'dataset': self.dataset.__dict__,
            'fine_tuning': self.fine_tuning.__dict__,
            'rag': self.rag.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'ResearchConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            dataset=DatasetConfig(**config_dict.get('dataset', {})),
            fine_tuning=FineTuningConfig(**config_dict.get('fine_tuning', {})),
            rag=RAGConfig(**config_dict.get('rag', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def setup_directories(self) -> None:
        """Create necessary directories for the experiment"""
        directories = [
            self.fine_tuning.output_dir,
            self.rag.vector_db_path,
            self.experiment.log_dir,
            self.experiment.results_dir,
            "./FineTuning/processed_data",
            "./RAG/processed_docs",
            "./RAG/results"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for computation"""
        if self.experiment.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.experiment.device)
    
    def set_random_seeds(self) -> None:
        """Set random seeds for reproducibility"""
        import random
        import numpy as np
        
        random.seed(self.experiment.random_seed)
        np.random.seed(self.experiment.random_seed)
        torch.manual_seed(self.experiment.random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.experiment.random_seed)
            torch.cuda.manual_seed_all(self.experiment.random_seed)
        
        if self.experiment.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

# Default configuration instance
default_config = ResearchConfig()

# Configuration validation
def validate_config(config: ResearchConfig) -> List[str]:
    """Validate configuration parameters and return list of issues"""
    issues = []
    
    # Dataset validation
    if config.dataset.sample_size <= 0:
        issues.append("Dataset sample_size must be positive")
    
    if not (0 < config.dataset.train_test_split < 1):
        issues.append("Dataset train_test_split must be between 0 and 1")
    
    # Fine-tuning validation
    if config.fine_tuning.lora_r <= 0:
        issues.append("LoRA rank (r) must be positive")
    
    if config.fine_tuning.learning_rate <= 0:
        issues.append("Learning rate must be positive")
    
    if config.fine_tuning.num_epochs <= 0:
        issues.append("Number of epochs must be positive")
    
    # RAG validation
    if config.rag.chunk_size <= config.rag.chunk_overlap:
        issues.append("Chunk size must be larger than chunk overlap")
    
    if config.rag.k_retrieve <= 0:
        issues.append("Number of documents to retrieve (k) must be positive")
    
    # Evaluation validation
    if not (0 < config.evaluation.significance_level < 1):
        issues.append("Significance level must be between 0 and 1")
    
    if len(config.evaluation.test_questions) < 5:
        issues.append("Need at least 5 test questions for robust evaluation")
    
    return issues

# Utility functions
def create_experiment_config(
    experiment_name: str,
    sample_size: int = 1000,
    num_epochs: int = 3,
    k_retrieve: int = 5
) -> ResearchConfig:
    """Create a custom configuration for specific experiments"""
    config = ResearchConfig()
    config.experiment.experiment_name = experiment_name
    config.dataset.sample_size = sample_size
    config.fine_tuning.num_epochs = num_epochs
    config.rag.k_retrieve = k_retrieve
    
    return config

def get_conference_config() -> ResearchConfig:
    """Get configuration optimized for conference paper results"""
    config = ResearchConfig()
    
    # Use larger sample for robust results
    config.dataset.sample_size = 2000
    
    # More thorough fine-tuning
    config.fine_tuning.num_epochs = 5
    config.fine_tuning.eval_steps = 25
    config.fine_tuning.save_steps = 50
    
    # More comprehensive RAG
    config.rag.k_retrieve = 7
    config.rag.max_context_length = 2000
    
    # Comprehensive evaluation
    config.evaluation.bootstrap_samples = 2000
    config.evaluation.test_questions.extend([
        "What are the legal remedies for contractual violations?",
        "How are intellectual property rights protected in agreements?",
        "What constitutes force majeure in legal contracts?",
        "How are arbitration clauses enforced in Indian law?",
        "What are the limitations of liability in commercial contracts?"
    ])
    
    return config

if __name__ == "__main__":
    # Example usage and validation
    config = default_config
    issues = validate_config(config)
    
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration is valid")
        
    # Save default configuration
    config.save_config("./configs/default_config.json")
    print("📁 Default configuration saved to ./configs/default_config.json")
    
    # Create conference configuration
    conference_config = get_conference_config()
    conference_config.save_config("./configs/conference_config.json")
    print("📊 Conference configuration saved to ./configs/conference_config.json") 