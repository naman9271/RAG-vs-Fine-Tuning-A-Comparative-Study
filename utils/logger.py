"""
Academic Research Logging System
Comprehensive logging for RAG vs Fine-Tuning comparative study
"""

import logging
import json
import time
import psutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import traceback

try:
    import pandas as pd
    import numpy as np
except ImportError:
    # Handle case where pandas/numpy aren't installed
    pd = None
    np = None

@dataclass
class ExperimentMetrics:
    """Container for experiment metrics"""
    experiment_id: str
    approach: str  # "fine_tuning" or "rag"
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Performance metrics
    total_runtime: Optional[float] = None
    memory_usage: Optional[List[float]] = None
    cpu_usage: Optional[List[float]] = None
    
    # Training metrics (for fine-tuning)
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    perplexity: Optional[float] = None
    training_time: Optional[float] = None
    
    # Evaluation metrics
    accuracy_scores: Optional[Dict[str, float]] = None
    quality_scores: Optional[Dict[str, float]] = None
    efficiency_scores: Optional[Dict[str, float]] = None
    
    # Configuration
    hyperparameters: Optional[Dict[str, Any]] = None
    dataset_info: Optional[Dict[str, Any]] = None
    
    # Results
    final_results: Optional[Dict[str, Any]] = None
    error_log: List[str] = field(default_factory=list)
    training_steps: List[Dict[str, Any]] = field(default_factory=list)

class AcademicLogger:
    """Comprehensive logging system for academic research"""
    
    def __init__(
        self, 
        experiment_name: str,
        log_dir: str = "./logs",
        log_level: str = "INFO",
        console_output: bool = True
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        
        # Setup logging
        self.logger = self._setup_logger(log_level, console_output)
        
        # Metrics tracking
        self.metrics = {}
        self.current_metrics = None
        
        # System monitoring
        self.monitor_resources = True
        self.resource_logs = []
        
        self.logger.info(f"🔬 Academic Logger initialized for experiment: {self.experiment_id}")
    
    def _setup_logger(self, log_level: str, console_output: bool) -> logging.Logger:
        """Setup comprehensive logging configuration"""
        
        # Create logger
        logger = logging.getLogger(self.experiment_id)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler for main log
        log_file = self.log_dir / f"{self.experiment_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        if console_output:
            # Simpler formatter for console
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def start_experiment(
        self, 
        approach: str,
        config: Dict[str, Any] = None,
        dataset_info: Dict[str, Any] = None
    ) -> str:
        """Start experiment tracking"""
        
        experiment_metrics = ExperimentMetrics(
            experiment_id=self.experiment_id,
            approach=approach,
            start_time=datetime.now(),
            hyperparameters=config,
            dataset_info=dataset_info,
            error_log=[]
        )
        
        self.current_metrics = experiment_metrics
        self.metrics[approach] = experiment_metrics
        
        self.logger.info(f"🚀 Starting {approach} experiment")
        self.logger.info(f"📊 Experiment ID: {self.experiment_id}")
        
        if config:
            self.logger.info(f"⚙️  Configuration: {json.dumps(config, indent=2)}")
        
        if dataset_info:
            self.logger.info(f"📚 Dataset info: {json.dumps(dataset_info, indent=2)}")
        
        # Log system information
        self._log_system_info()
        
        return self.experiment_id
    
    def log_training_step(
        self, 
        step: int, 
        loss: float, 
        learning_rate: float = None,
        additional_metrics: Dict[str, float] = None
    ):
        """Log training step information"""
        
        log_msg = f"📈 Step {step}: Loss = {loss:.4f}"
        
        if learning_rate:
            log_msg += f", LR = {learning_rate:.2e}"
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                log_msg += f", {key} = {value:.4f}"
        
        self.logger.info(log_msg)
        
        # Store in metrics
        if self.current_metrics:
            if not hasattr(self.current_metrics, 'training_steps'):
                self.current_metrics.training_steps = []
            
            step_data = {
                'step': step,
                'loss': loss,
                'learning_rate': learning_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            if additional_metrics:
                step_data.update(additional_metrics)
            
            self.current_metrics.training_steps.append(step_data)
    
    def log_evaluation_results(
        self, 
        results: Dict[str, Any],
        phase: str = "evaluation"
    ):
        """Log evaluation results"""
        
        self.logger.info(f"📊 {phase.title()} Results:")
        
        for key, value in results.items():
            if isinstance(value, (int, float)):
                if key.endswith('_time'):
                    self.logger.info(f"   ⏱️  {key}: {value:.3f}s")
                elif key.endswith('_score'):
                    self.logger.info(f"   🎯 {key}: {value:.4f}")
                else:
                    self.logger.info(f"   📈 {key}: {value:.4f}")
            else:
                self.logger.info(f"   📋 {key}: {value}")
        
        # Store in current metrics
        if self.current_metrics:
            if phase == "evaluation":
                self.current_metrics.final_results = results
            elif phase == "quality":
                self.current_metrics.quality_scores = results
            elif phase == "efficiency":
                self.current_metrics.efficiency_scores = results
    
    def log_performance_metrics(
        self, 
        response_times: List[float],
        memory_usage: Optional[List[float]] = None,
        gpu_usage: Optional[List[float]] = None
    ):
        """Log performance metrics"""
        
        # Check if numpy is available
        if np is None:
            self.logger.warning("NumPy not available, skipping performance metrics")
            return
        
        # Response time statistics
        rt_stats = {
            'mean': np.mean(response_times),
            'std': np.std(response_times),
            'min': np.min(response_times),
            'max': np.max(response_times),
            'p95': np.percentile(response_times, 95),
            'p99': np.percentile(response_times, 99)
        }
        
        self.logger.info("⚡ Performance Metrics:")
        self.logger.info(f"   Response Time - Mean: {rt_stats['mean']:.3f}s ± {rt_stats['std']:.3f}s")
        self.logger.info(f"   Response Time - P95: {rt_stats['p95']:.3f}s, P99: {rt_stats['p99']:.3f}s")
        
        if memory_usage:
            mem_stats = {
                'mean': np.mean(memory_usage),
                'max': np.max(memory_usage)
            }
            self.logger.info(f"   Memory Usage - Mean: {mem_stats['mean']:.2f}GB, Peak: {mem_stats['max']:.2f}GB")
        
        if gpu_usage:
            gpu_stats = {
                'mean': np.mean(gpu_usage),
                'max': np.max(gpu_usage)
            }
            self.logger.info(f"   GPU Usage - Mean: {gpu_stats['mean']:.1f}%, Peak: {gpu_stats['max']:.1f}%")
        
        # Store in metrics
        if self.current_metrics:
            self.current_metrics.efficiency_scores = {
                'response_time_stats': rt_stats,
                'memory_stats': mem_stats if memory_usage else None,
                'gpu_stats': gpu_stats if gpu_usage else None
            }
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with full traceback"""
        
        error_msg = f"❌ Error in {context}: {str(error)}"
        self.logger.error(error_msg)
        
        # Log full traceback
        tb_str = traceback.format_exc()
        self.logger.error(f"Traceback:\n{tb_str}")
        
        # Store in metrics
        if self.current_metrics:
            self.current_metrics.error_log.append({
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'error': str(error),
                'traceback': tb_str
            })
    
    def log_comparison_results(
        self, 
        approach1_results: Dict[str, Any],
        approach2_results: Dict[str, Any],
        statistical_tests: Dict[str, Any] = None
    ):
        """Log comparison results between approaches"""
        
        self.logger.info("🔬 Comparative Analysis Results:")
        self.logger.info("=" * 50)
        
        # Extract approach names
        approach1 = approach1_results.get('approach', 'Approach 1')
        approach2 = approach2_results.get('approach', 'Approach 2')
        
        self.logger.info(f"📊 Comparing {approach1} vs {approach2}")
        
        # Compare key metrics
        key_metrics = ['avg_response_time', 'accuracy_score', 'quality_score']
        
        for metric in key_metrics:
            if metric in approach1_results and metric in approach2_results:
                val1 = approach1_results[metric]
                val2 = approach2_results[metric]
                diff = val2 - val1
                percent_diff = (diff / val1) * 100 if val1 != 0 else float('inf')
                
                winner = approach2 if diff > 0 else approach1
                self.logger.info(f"   {metric}: {val1:.3f} vs {val2:.3f} (Δ{diff:+.3f}, {percent_diff:+.1f}%) → {winner}")
        
        # Statistical significance
        if statistical_tests:
            self.logger.info("\n🔬 Statistical Significance:")
            for test_name, test_result in statistical_tests.items():
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    p_val = test_result['p_value']
                    significant = test_result.get('significant', p_val < 0.05)
                    self.logger.info(f"   {test_name}: p = {p_val:.4f} ({'significant' if significant else 'not significant'})")
    
    def finish_experiment(self, final_results: Dict[str, Any] = None):
        """Finish experiment and save all logs"""
        
        if self.current_metrics:
            self.current_metrics.end_time = datetime.now()
            self.current_metrics.total_runtime = (
                self.current_metrics.end_time - self.current_metrics.start_time
            ).total_seconds()
            
            if final_results:
                self.current_metrics.final_results = final_results
        
        self.logger.info(f"🏁 Experiment {self.current_metrics.approach} completed")
        self.logger.info(f"⏱️  Total runtime: {self.current_metrics.total_runtime:.2f} seconds")
        
        # Save detailed metrics
        self._save_experiment_metrics()
        
        # Generate summary report
        self._generate_experiment_summary()
    
    def _log_system_info(self):
        """Log system information for reproducibility"""
        
        self.logger.info("💻 System Information:")
        self.logger.info(f"   Python: {sys.version}")
        
        try:
            import torch
            import transformers
            import numpy
            import pandas
            
            self.logger.info(f"   PyTorch: {torch.__version__}")
            self.logger.info(f"   Transformers: {transformers.__version__}")
            self.logger.info(f"   NumPy: {numpy.__version__}")
            self.logger.info(f"   Pandas: {pandas.__version__}")
            
            # Hardware information
            self.logger.info(f"   CPU: {psutil.cpu_count()} cores")
            self.logger.info(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
            
            if torch.cuda.is_available():
                self.logger.info(f"   GPU: {torch.cuda.get_device_name()}")
                self.logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            else:
                self.logger.info("   GPU: Not available")
                
        except ImportError as e:
            self.logger.warning(f"Could not import some packages: {e}")
    
    def _save_experiment_metrics(self):
        """Save detailed experiment metrics to JSON"""
        
        if not self.current_metrics:
            return
        
        # Convert to dictionary
        metrics_dict = asdict(self.current_metrics)
        
        # Handle datetime serialization
        if metrics_dict['start_time']:
            metrics_dict['start_time'] = metrics_dict['start_time'].isoformat()
        if metrics_dict['end_time']:
            metrics_dict['end_time'] = metrics_dict['end_time'].isoformat()
        
        # Save to file
        metrics_file = self.log_dir / f"{self.experiment_id}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        self.logger.info(f"💾 Experiment metrics saved to: {metrics_file}")
    
    def _generate_experiment_summary(self):
        """Generate experiment summary report"""
        
        if not self.current_metrics:
            return
        
        summary_file = self.log_dir / f"{self.experiment_id}_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"# Experiment Summary: {self.experiment_id}\n\n")
            f.write(f"**Approach**: {self.current_metrics.approach}\n")
            f.write(f"**Start Time**: {self.current_metrics.start_time}\n")
            f.write(f"**End Time**: {self.current_metrics.end_time}\n")
            f.write(f"**Total Runtime**: {self.current_metrics.total_runtime:.2f} seconds\n\n")
            
            if self.current_metrics.hyperparameters:
                f.write("## Configuration\n")
                f.write("```json\n")
                f.write(json.dumps(self.current_metrics.hyperparameters, indent=2))
                f.write("\n```\n\n")
            
            if self.current_metrics.final_results:
                f.write("## Final Results\n")
                for key, value in self.current_metrics.final_results.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            if self.current_metrics.error_log:
                f.write("## Errors Encountered\n")
                for error in self.current_metrics.error_log:
                    f.write(f"- {error['timestamp']}: {error['error']}\n")
        
        self.logger.info(f"📄 Experiment summary saved to: {summary_file}")

class ExperimentTracker:
    """High-level experiment tracking for comparative studies"""
    
    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments = {}
        self.current_study = None
    
    def start_comparative_study(self, study_name: str) -> str:
        """Start a new comparative study"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_id = f"{study_name}_{timestamp}"
        
        study_dir = self.base_dir / study_id
        study_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_study = {
            'id': study_id,
            'name': study_name,
            'dir': study_dir,
            'start_time': datetime.now(),
            'experiments': {},
            'comparisons': []
        }
        
        return study_id
    
    def add_experiment(self, approach: str, config: Dict[str, Any] = None) -> AcademicLogger:
        """Add an experiment to the current study"""
        
        if not self.current_study:
            raise ValueError("No active study. Call start_comparative_study() first.")
        
        # Create logger for this experiment
        log_dir = self.current_study['dir'] / "logs"
        logger = AcademicLogger(
            experiment_name=f"{self.current_study['name']}_{approach}",
            log_dir=str(log_dir)
        )
        
        self.current_study['experiments'][approach] = {
            'logger': logger,
            'config': config,
            'results': None
        }
        
        return logger
    
    def finalize_study(self, comparison_results: Dict[str, Any] = None):
        """Finalize comparative study and generate final report"""
        
        if not self.current_study:
            return
        
        # Save study metadata
        study_file = self.current_study['dir'] / "study_metadata.json"
        study_data = {
            'study_id': self.current_study['id'],
            'study_name': self.current_study['name'],
            'start_time': self.current_study['start_time'].isoformat(),
            'end_time': datetime.now().isoformat(),
            'experiments': list(self.current_study['experiments'].keys()),
            'comparison_results': comparison_results
        }
        
        with open(study_file, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        print(f"📊 Comparative study completed: {self.current_study['id']}")
        print(f"📁 Results saved in: {self.current_study['dir']}")

# Utility functions
def setup_academic_logging(experiment_name: str, config: Dict[str, Any] = None) -> AcademicLogger:
    """Quick setup for academic logging"""
    
    logger = AcademicLogger(experiment_name)
    
    if config:
        logger.logger.info("⚙️  Configuration loaded:")
        for key, value in config.items():
            logger.logger.info(f"   {key}: {value}")
    
    return logger

def log_comparison_table(
    logger: AcademicLogger,
    results1: Dict[str, Any],
    results2: Dict[str, Any],
    approach1_name: str = "Approach 1",
    approach2_name: str = "Approach 2"
):
    """Log a formatted comparison table"""
    
    logger.logger.info(f"📊 Comparison: {approach1_name} vs {approach2_name}")
    logger.logger.info("=" * 70)
    logger.logger.info(f"{'Metric':<25} {'Approach 1':<15} {'Approach 2':<15} {'Winner':<15}")
    logger.logger.info("-" * 70)
    
    # Find common metrics
    common_metrics = set(results1.keys()).intersection(set(results2.keys()))
    
    for metric in sorted(common_metrics):
        val1 = results1[metric]
        val2 = results2[metric]
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            winner = approach2_name if val2 > val1 else approach1_name
            logger.logger.info(f"{metric:<25} {val1:<15.3f} {val2:<15.3f} {winner:<15}")

if __name__ == "__main__":
    # Example usage
    logger = setup_academic_logging("test_experiment")
    logger.logger.info("🧪 Academic logging system test completed")
    print("✅ Academic Logger is ready for use") 