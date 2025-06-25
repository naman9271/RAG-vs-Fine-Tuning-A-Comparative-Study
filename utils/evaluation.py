"""
Academic Evaluation Framework for RAG vs Fine-Tuning Comparison
Comprehensive metrics and statistical analysis for conference paper
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import json
import time
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Evaluation metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("⚠️  BERTScore not available. Install with: pip install bert-score")

# Statistical analysis
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    approach: str  # "fine_tuning" or "rag"
    metrics: Dict[str, float]
    response_times: List[float]
    memory_usage: List[float]
    responses: List[str]
    questions: List[str]
    contexts: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    metadata: Dict[str, Any] = None

class AcademicEvaluator:
    """Comprehensive evaluation framework for academic research"""
    
    def __init__(self, config=None):
        self.config = config
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        self.bleu_smoother = SmoothingFunction().method1
        
    def evaluate_response_quality(
        self, 
        responses: List[str], 
        references: List[str] = None,
        contexts: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate response quality using multiple metrics
        
        Args:
            responses: Generated responses
            references: Reference answers (if available)
            contexts: Source contexts for RAG evaluation
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Basic text quality metrics
        metrics['avg_response_length'] = np.mean([len(r) for r in responses])
        metrics['response_length_std'] = np.std([len(r) for r in responses])
        metrics['avg_word_count'] = np.mean([len(r.split()) for r in responses])
        
        # Coherence metrics (simple heuristics)
        metrics['avg_sentences_per_response'] = np.mean([
            len([s for s in r.split('.') if len(s.strip()) > 0]) 
            for r in responses
        ])
        
        # Legal-specific metrics
        legal_terms = [
            'contract', 'agreement', 'provision', 'clause', 'obligation',
            'liability', 'breach', 'remedy', 'damages', 'court', 'section',
            'act', 'law', 'legal', 'jurisdiction', 'arbitration'
        ]
        
        legal_term_coverage = []
        for response in responses:
            response_lower = response.lower()
            coverage = sum(1 for term in legal_terms if term in response_lower)
            legal_term_coverage.append(coverage / len(legal_terms))
        
        metrics['legal_term_coverage'] = np.mean(legal_term_coverage)
        metrics['legal_term_coverage_std'] = np.std(legal_term_coverage)
        
        # Reference-based metrics (if references available)
        if references:
            rouge_scores = self._compute_rouge_scores(responses, references)
            metrics.update(rouge_scores)
            
            bleu_scores = self._compute_bleu_scores(responses, references)
            metrics.update(bleu_scores)
            
            if BERT_SCORE_AVAILABLE:
                bert_scores = self._compute_bert_scores(responses, references)
                metrics.update(bert_scores)
        
        # Context utilization (for RAG)
        if contexts:
            context_metrics = self._evaluate_context_utilization(responses, contexts)
            metrics.update(context_metrics)
            
        return metrics
    
    def _compute_rouge_scores(
        self, 
        responses: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE scores"""
        rouge_scores = {
            'rouge1_precision': [], 'rouge1_recall': [], 'rouge1_f1': [],
            'rouge2_precision': [], 'rouge2_recall': [], 'rouge2_f1': [],
            'rougeL_precision': [], 'rougeL_recall': [], 'rougeL_f1': []
        }
        
        for response, reference in zip(responses, references):
            scores = self.rouge_scorer.score(reference, response)
            
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                rouge_scores[f'{metric}_precision'].append(scores[metric].precision)
                rouge_scores[f'{metric}_recall'].append(scores[metric].recall)
                rouge_scores[f'{metric}_f1'].append(scores[metric].fmeasure)
        
        # Average scores
        averaged_scores = {}
        for key, values in rouge_scores.items():
            averaged_scores[f'avg_{key}'] = np.mean(values)
            averaged_scores[f'std_{key}'] = np.std(values)
            
        return averaged_scores
    
    def _compute_bleu_scores(
        self, 
        responses: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Compute BLEU scores"""
        bleu_scores = []
        
        for response, reference in zip(responses, references):
            # Tokenize
            response_tokens = response.split()
            reference_tokens = [reference.split()]  # BLEU expects list of references
            
            # Compute BLEU score
            bleu = sentence_bleu(
                reference_tokens, 
                response_tokens, 
                smoothing_function=self.bleu_smoother
            )
            bleu_scores.append(bleu)
        
        return {
            'avg_bleu_score': np.mean(bleu_scores),
            'std_bleu_score': np.std(bleu_scores),
            'bleu_scores': bleu_scores
        }
    
    def _compute_bert_scores(
        self, 
        responses: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Compute BERTScore"""
        if not BERT_SCORE_AVAILABLE:
            return {}
            
        P, R, F1 = bert_score(responses, references, lang="en", verbose=False)
        
        return {
            'avg_bert_precision': P.mean().item(),
            'avg_bert_recall': R.mean().item(),
            'avg_bert_f1': F1.mean().item(),
            'std_bert_precision': P.std().item(),
            'std_bert_recall': R.std().item(),
            'std_bert_f1': F1.std().item()
        }
    
    def _evaluate_context_utilization(
        self, 
        responses: List[str], 
        contexts: List[str]
    ) -> Dict[str, float]:
        """Evaluate how well responses utilize provided context"""
        utilization_scores = []
        
        for response, context in zip(responses, contexts):
            # Simple overlap-based metric
            response_words = set(response.lower().split())
            context_words = set(context.lower().split())
            
            if len(context_words) > 0:
                overlap = len(response_words.intersection(context_words))
                utilization = overlap / len(context_words)
                utilization_scores.append(min(utilization, 1.0))
            else:
                utilization_scores.append(0.0)
        
        return {
            'avg_context_utilization': np.mean(utilization_scores),
            'std_context_utilization': np.std(utilization_scores)
        }
    
    def evaluate_efficiency(
        self, 
        response_times: List[float],
        memory_usage: List[float] = None,
        training_time: float = None
    ) -> Dict[str, float]:
        """Evaluate computational efficiency"""
        metrics = {
            'avg_response_time': np.mean(response_times),
            'std_response_time': np.std(response_times),
            'min_response_time': np.min(response_times),
            'max_response_time': np.max(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99)
        }
        
        if memory_usage:
            metrics.update({
                'avg_memory_usage': np.mean(memory_usage),
                'std_memory_usage': np.std(memory_usage),
                'peak_memory_usage': np.max(memory_usage)
            })
        
        if training_time:
            metrics['training_time'] = training_time
            
        return metrics
    
    def evaluate_interpretability(
        self, 
        approach: str,
        sources: List[str] = None,
        contexts: List[str] = None
    ) -> Dict[str, float]:
        """Evaluate interpretability aspects"""
        metrics = {}
        
        if approach == "rag":
            # RAG interpretability metrics
            metrics['source_attribution_rate'] = 1.0 if sources else 0.0
            metrics['context_transparency'] = 1.0 if contexts else 0.0
            
            if sources:
                # Source diversity
                unique_sources = len(set(sources))
                total_sources = len(sources)
                metrics['source_diversity'] = unique_sources / total_sources if total_sources > 0 else 0
                
            if contexts:
                # Context quality metrics
                avg_context_length = np.mean([len(c) for c in contexts])
                metrics['avg_context_length'] = avg_context_length
                
        elif approach == "fine_tuning":
            # Fine-tuning interpretability (limited)
            metrics['source_attribution_rate'] = 0.0
            metrics['context_transparency'] = 0.0
            metrics['model_interpretability'] = 0.1  # Very limited
            
        return metrics
    
    def statistical_comparison(
        self, 
        results1: EvaluationResult,
        results2: EvaluationResult,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison between two approaches
        
        Args:
            results1: Results from first approach
            results2: Results from second approach
            alpha: Significance level
            
        Returns:
            Statistical comparison results
        """
        comparison = {
            'approach1': results1.approach,
            'approach2': results2.approach,
            'significance_level': alpha,
            'tests': {}
        }
        
        # Compare response times
        if results1.response_times and results2.response_times:
            time_comparison = self._compare_distributions(
                results1.response_times,
                results2.response_times,
                "response_time",
                alpha
            )
            comparison['tests']['response_time'] = time_comparison
        
        # Compare quality metrics
        common_metrics = set(results1.metrics.keys()).intersection(
            set(results2.metrics.keys())
        )
        
        for metric in common_metrics:
            if isinstance(results1.metrics[metric], (int, float)) and \
               isinstance(results2.metrics[metric], (int, float)):
                # For single values, we can't do statistical tests
                comparison['tests'][metric] = {
                    'approach1_value': results1.metrics[metric],
                    'approach2_value': results2.metrics[metric],
                    'difference': results2.metrics[metric] - results1.metrics[metric],
                    'relative_improvement': (
                        (results2.metrics[metric] - results1.metrics[metric]) / 
                        results1.metrics[metric] * 100
                    ) if results1.metrics[metric] != 0 else float('inf')
                }
        
        return comparison
    
    def _compare_distributions(
        self, 
        data1: List[float], 
        data2: List[float],
        metric_name: str,
        alpha: float
    ) -> Dict[str, Any]:
        """Compare two distributions statistically"""
        
        # Descriptive statistics
        stats1 = {
            'mean': np.mean(data1),
            'std': np.std(data1),
            'median': np.median(data1),
            'min': np.min(data1),
            'max': np.max(data1)
        }
        
        stats2 = {
            'mean': np.mean(data2),
            'std': np.std(data2),
            'median': np.median(data2),
            'min': np.min(data2),
            'max': np.max(data2)
        }
        
        # Statistical tests
        # Mann-Whitney U test (non-parametric)
        mw_statistic, mw_pvalue = mannwhitneyu(data1, data2, alternative='two-sided')
        
        # T-test (parametric)
        t_statistic, t_pvalue = ttest_ind(data1, data2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1) + 
                             (len(data2) - 1) * np.var(data2)) / 
                            (len(data1) + len(data2) - 2))
        cohens_d = (np.mean(data2) - np.mean(data1)) / pooled_std if pooled_std > 0 else 0
        
        return {
            'metric': metric_name,
            'group1_stats': stats1,
            'group2_stats': stats2,
            'statistical_tests': {
                'mann_whitney_u': {
                    'statistic': mw_statistic,
                    'p_value': mw_pvalue,
                    'significant': mw_pvalue < alpha
                },
                't_test': {
                    'statistic': t_statistic,
                    'p_value': t_pvalue,
                    'significant': t_pvalue < alpha
                }
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'magnitude': self._interpret_cohens_d(cohens_d)
            }
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_comparison_report(
        self, 
        results1: EvaluationResult,
        results2: EvaluationResult,
        save_path: str = None
    ) -> str:
        """Generate comprehensive comparison report"""
        
        # Statistical comparison
        stats_comparison = self.statistical_comparison(results1, results2)
        
        # Generate report
        report = []
        report.append("# RAG vs Fine-Tuning: Comparative Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Executive summary
        report.append("## Executive Summary")
        report.append(f"**Approach 1**: {results1.approach}")
        report.append(f"**Approach 2**: {results2.approach}")
        report.append("")
        
        # Performance comparison
        report.append("## Performance Metrics Comparison")
        report.append("")
        report.append("| Metric | {} | {} | Difference | Winner |".format(
            results1.approach.title(), results2.approach.title()
        ))
        report.append("|--------|" + "--------|" * 4)
        
        # Add metric comparisons
        for test_name, test_results in stats_comparison['tests'].items():
            if 'approach1_value' in test_results:
                val1 = test_results['approach1_value']
                val2 = test_results['approach2_value']
                diff = test_results['difference']
                winner = results2.approach if diff > 0 else results1.approach
                
                report.append(f"| {test_name} | {val1:.3f} | {val2:.3f} | {diff:.3f} | {winner} |")
        
        report.append("")
        
        # Statistical significance
        report.append("## Statistical Significance")
        report.append("")
        
        for test_name, test_results in stats_comparison['tests'].items():
            if 'statistical_tests' in test_results:
                mw_test = test_results['statistical_tests']['mann_whitney_u']
                report.append(f"**{test_name}**: p-value = {mw_test['p_value']:.4f} "
                             f"({'significant' if mw_test['significant'] else 'not significant'})")
        
        report.append("")
        
        # Conclusions
        report.append("## Key Findings")
        report.append("")
        
        # Determine overall winner based on multiple factors
        winner_factors = []
        if 'response_time' in stats_comparison['tests']:
            rt_test = stats_comparison['tests']['response_time']
            if rt_test['group1_stats']['mean'] < rt_test['group2_stats']['mean']:
                winner_factors.append(f"{results1.approach} is faster")
            else:
                winner_factors.append(f"{results2.approach} is faster")
        
        for factor in winner_factors:
            report.append(f"- {factor}")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {save_path}")
        
        return report_text
    
    def create_visualization(
        self, 
        results1: EvaluationResult,
        results2: EvaluationResult,
        save_path: str = None
    ) -> None:
        """Create visualization comparing both approaches"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RAG vs Fine-Tuning: Performance Comparison', fontsize=16, fontweight='bold')
        
        # Response time comparison
        if results1.response_times and results2.response_times:
            axes[0, 0].hist(results1.response_times, alpha=0.7, label=results1.approach, bins=20)
            axes[0, 0].hist(results2.response_times, alpha=0.7, label=results2.approach, bins=20)
            axes[0, 0].set_title('Response Time Distribution')
            axes[0, 0].set_xlabel('Time (seconds)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
        
        # Metric comparison radar chart (if available)
        common_metrics = ['avg_response_time', 'legal_term_coverage', 'avg_context_utilization']
        available_metrics = []
        values1, values2 = [], []
        
        for metric in common_metrics:
            if metric in results1.metrics and metric in results2.metrics:
                available_metrics.append(metric)
                values1.append(results1.metrics[metric])
                values2.append(results2.metrics[metric])
        
        if available_metrics:
            x_pos = np.arange(len(available_metrics))
            width = 0.35
            
            axes[0, 1].bar(x_pos - width/2, values1, width, label=results1.approach, alpha=0.7)
            axes[0, 1].bar(x_pos + width/2, values2, width, label=results2.approach, alpha=0.7)
            axes[0, 1].set_title('Metric Comparison')
            axes[0, 1].set_xlabel('Metrics')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(available_metrics, rotation=45)
            axes[0, 1].legend()
        
        # Memory usage (if available)
        if results1.memory_usage and results2.memory_usage:
            box_data = [results1.memory_usage, results2.memory_usage]
            box_labels = [results1.approach, results2.approach]
            axes[0, 2].boxplot(box_data, labels=box_labels)
            axes[0, 2].set_title('Memory Usage Distribution')
            axes[0, 2].set_ylabel('Memory (GB)')
        
        # Quality metrics comparison
        quality_metrics = ['avg_rouge1_f1', 'avg_bleu_score', 'avg_bert_f1']
        quality_values1, quality_values2 = [], []
        quality_labels = []
        
        for metric in quality_metrics:
            if metric in results1.metrics and metric in results2.metrics:
                quality_labels.append(metric.replace('avg_', '').replace('_', ' ').title())
                quality_values1.append(results1.metrics[metric])
                quality_values2.append(results2.metrics[metric])
        
        if quality_labels:
            x_pos = np.arange(len(quality_labels))
            axes[1, 0].plot(x_pos, quality_values1, 'o-', label=results1.approach, linewidth=2)
            axes[1, 0].plot(x_pos, quality_values2, 's-', label=results2.approach, linewidth=2)
            axes[1, 0].set_title('Quality Metrics')
            axes[1, 0].set_xlabel('Metrics')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(quality_labels, rotation=45)
            axes[1, 0].legend()
        
        # Interpretability comparison
        interpretability_metrics = ['source_attribution_rate', 'context_transparency']
        interp_values1, interp_values2 = [], []
        interp_labels = []
        
        for metric in interpretability_metrics:
            if metric in results1.metrics and metric in results2.metrics:
                interp_labels.append(metric.replace('_', ' ').title())
                interp_values1.append(results1.metrics[metric])
                interp_values2.append(results2.metrics[metric])
        
        if interp_labels:
            x_pos = np.arange(len(interp_labels))
            width = 0.35
            axes[1, 1].bar(x_pos - width/2, interp_values1, width, label=results1.approach, alpha=0.7)
            axes[1, 1].bar(x_pos + width/2, interp_values2, width, label=results2.approach, alpha=0.7)
            axes[1, 1].set_title('Interpretability Metrics')
            axes[1, 1].set_xlabel('Metrics')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(interp_labels, rotation=45)
            axes[1, 1].legend()
        
        # Overall comparison summary
        axes[1, 2].text(0.1, 0.8, f"Approaches Compared:", fontsize=12, fontweight='bold')
        axes[1, 2].text(0.1, 0.7, f"• {results1.approach}", fontsize=10)
        axes[1, 2].text(0.1, 0.6, f"• {results2.approach}", fontsize=10)
        axes[1, 2].text(0.1, 0.4, f"Questions Evaluated: {len(results1.questions)}", fontsize=10)
        axes[1, 2].text(0.1, 0.3, f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", fontsize=10)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Evaluation Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        
        plt.show()

# Utility functions for academic evaluation
def bootstrap_confidence_interval(
    data: List[float], 
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval"""
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return ci_lower, ci_upper

def calculate_practical_significance(
    effect_size: float,
    threshold: float = 0.1
) -> bool:
    """Determine if difference is practically significant"""
    return abs(effect_size) >= threshold

def format_results_for_paper(results: Dict[str, Any]) -> str:
    """Format results for academic paper"""
    formatted = []
    
    for key, value in results.items():
        if isinstance(value, float):
            if key.endswith('_time'):
                formatted.append(f"{key}: {value:.2f}s")
            elif key.endswith('_score'):
                formatted.append(f"{key}: {value:.3f}")
            else:
                formatted.append(f"{key}: {value:.3f}")
        else:
            formatted.append(f"{key}: {value}")
    
    return "\n".join(formatted)

if __name__ == "__main__":
    # Example usage
    evaluator = AcademicEvaluator()
    print("📊 Academic Evaluation Framework initialized")
    print("✅ Ready for comprehensive RAG vs Fine-Tuning comparison") 