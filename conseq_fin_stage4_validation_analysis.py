#!/usr/bin/env python3
"""
Analyze validation results from cluster assignment accuracy tests

This script processes the evaluation results from all three validation types
and generates comprehensive metrics and reports.

Usage:
    python conseq_fin_stage4_validation_analysis.py
    python conseq_fin_stage4_validation_analysis.py --logs-dir custom_logs/
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_stage4_validation_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ValidationAnalyzer:
    def __init__(self, logs_dir: str = 'logs'):
        self.logs_dir = Path(logs_dir)
        self.results = {}
        
    def load_validation_results(self, validation_type: str) -> pd.DataFrame:
        """Load results for a specific validation type"""
        # Find the most recent eval file for this validation type
        pattern = f'*validate-{validation_type.replace("_", "-")}*.eval'
        eval_files = list(self.logs_dir.glob(pattern))
        
        if not eval_files:
            logger.warning(f"No evaluation files found for {validation_type}")
            return pd.DataFrame()
        
        # Get the most recent file
        eval_file = max(eval_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading {validation_type} results from: {eval_file.name}")
        
        # Create temporary directory for analysis
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / eval_file.name
        
        try:
            shutil.copy2(eval_file, temp_file)
            
            from inspect_ai.analysis.beta import evals_df
            
            # Load evaluation data
            df = evals_df(str(temp_file))
            
            # Extract relevant columns
            results_df = pd.DataFrame({
                'sample_id': df.index,
                'score': df.get('score', 0),
                'target': df.get('target', ''),
                'output': df.get('output', ''),
                'metadata': df.get('metadata', {})
            })
            
            return results_df
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def calculate_metrics(self, results_df: pd.DataFrame, validation_type: str) -> Dict[str, Any]:
        """Calculate accuracy metrics for a validation type"""
        if results_df.empty:
            return {}
        
        # Basic accuracy
        accuracy = results_df['score'].mean()
        n_samples = len(results_df)
        n_correct = (results_df['score'] == 1.0).sum()
        
        metrics = {
            'validation_type': validation_type,
            'accuracy': accuracy,
            'n_samples': n_samples,
            'n_correct': n_correct,
            'n_incorrect': n_samples - n_correct
        }
        
        # Additional metrics based on validation type
        if validation_type == 'l3_to_l2':
            # For L3->L2, analyze which clusters are most confused
            incorrect = results_df[results_df['score'] < 1.0]
            if len(incorrect) > 0:
                # This would require parsing the output to see which clusters were chosen
                metrics['error_rate_by_l1'] = self._analyze_l3_l2_errors(results_df)
        
        elif validation_type == 'l2_to_l1':
            # For L2->L1, this should have very high accuracy
            metrics['perfect_accuracy'] = (accuracy == 1.0)
            
        elif validation_type == 'l3_to_l1':
            # For L3->L1, compare to two-step accuracy
            metrics['direct_assignment_accuracy'] = accuracy
        
        return metrics
    
    def _analyze_l3_l2_errors(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze error patterns in L3->L2 validation"""
        error_analysis = {}
        
        # Group errors by Level 1 cluster (from metadata)
        for _, row in results_df.iterrows():
            if row['score'] < 1.0 and isinstance(row['metadata'], dict):
                l1_cluster = row['metadata'].get('correct_l1', 'unknown')
                if l1_cluster not in error_analysis:
                    error_analysis[l1_cluster] = {'errors': 0, 'total': 0}
                error_analysis[l1_cluster]['errors'] += 1
            
            # Count total per L1
            if isinstance(row['metadata'], dict):
                l1_cluster = row['metadata'].get('correct_l1', 'unknown')
                if l1_cluster not in error_analysis:
                    error_analysis[l1_cluster] = {'errors': 0, 'total': 0}
                error_analysis[l1_cluster]['total'] += 1
        
        # Calculate error rates
        error_rates = {}
        for l1, counts in error_analysis.items():
            if counts['total'] > 0:
                error_rates[l1] = counts['errors'] / counts['total']
        
        return error_rates
    
    def create_confusion_analysis(self, results_df: pd.DataFrame, validation_type: str):
        """Create confusion matrix or similar analysis"""
        # This would require parsing the actual vs predicted clusters
        # For now, we'll create a summary
        
        if validation_type in ['l2_to_l1', 'l3_to_l1']:
            # These have only 10 possible answers, so we can create a confusion matrix
            # Would need to parse outputs to get predicted values
            pass
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'validation_results': {},
            'summary': {}
        }
        
        # Process each validation type
        validation_types = ['l3_to_l2', 'l2_to_l1', 'l3_to_l1']
        
        for val_type in validation_types:
            logger.info(f"\nAnalyzing {val_type} validation...")
            
            # Load results
            results_df = self.load_validation_results(val_type)
            
            if not results_df.empty:
                # Calculate metrics
                metrics = self.calculate_metrics(results_df, val_type)
                report['validation_results'][val_type] = metrics
                
                # Log key findings
                logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
                logger.info(f"  Samples: {metrics['n_samples']}")
                logger.info(f"  Correct: {metrics['n_correct']}")
        
        # Overall summary
        if report['validation_results']:
            report['summary'] = {
                'l3_to_l1_accuracy': report['validation_results'].get('l3_to_l1', {}).get('accuracy', 0),
                'l2_to_l1_accuracy': report['validation_results'].get('l2_to_l1', {}).get('accuracy', 0),
                'l3_to_l2_accuracy': report['validation_results'].get('l3_to_l2', {}).get('accuracy', 0),
                'hierarchy_quality': self._assess_hierarchy_quality(report['validation_results'])
            }
        
        # Save report
        report_file = 'conseq_fin_stage4_validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nSaved validation report to {report_file}")
        
        return report
    
    def _assess_hierarchy_quality(self, results: Dict) -> str:
        """Assess overall hierarchy quality based on validation results"""
        l2_l1_acc = results.get('l2_to_l1', {}).get('accuracy', 0)
        l3_l1_acc = results.get('l3_to_l1', {}).get('accuracy', 0)
        l3_l2_acc = results.get('l3_to_l2', {}).get('accuracy', 0)
        
        if l2_l1_acc > 0.95 and l3_l1_acc > 0.9:
            return "Excellent - Clear hierarchical structure"
        elif l2_l1_acc > 0.9 and l3_l1_acc > 0.8:
            return "Good - Well-defined hierarchy with minor ambiguities"
        elif l2_l1_acc > 0.8 and l3_l1_acc > 0.7:
            return "Fair - Some cluster overlap, consider refinement"
        else:
            return "Poor - Significant cluster confusion, needs restructuring"
    
    def create_visualizations(self):
        """Create visualization plots for validation results"""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Accuracy comparison bar chart
        plt.figure(figsize=(10, 6))
        
        val_types = []
        accuracies = []
        
        for val_type, metrics in self.results.items():
            if 'accuracy' in metrics:
                val_types.append(val_type.replace('_', '->').upper())
                accuracies.append(metrics['accuracy'])
        
        plt.bar(val_types, accuracies)
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.title('Cluster Assignment Validation Accuracy')
        plt.axhline(y=0.9, color='g', linestyle='--', label='90% target')
        plt.axhline(y=0.8, color='y', linestyle='--', label='80% threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('conseq_fin_stage4_validation_accuracy.png')
        plt.close()
        
        logger.info("Saved visualization to conseq_fin_stage4_validation_accuracy.png")

def main():
    parser = argparse.ArgumentParser(description='Analyze cluster validation results')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory containing evaluation logs')
    
    args = parser.parse_args()
    
    logger.info("Starting validation analysis...")
    
    analyzer = ValidationAnalyzer(logs_dir=args.logs_dir)
    
    # Generate report
    report = analyzer.generate_report()
    
    # Create visualizations
    analyzer.results = report.get('validation_results', {})
    analyzer.create_visualizations()
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*50)
    
    if 'summary' in report:
        logger.info(f"L3→L1 Direct Accuracy: {report['summary']['l3_to_l1_accuracy']:.1%}")
        logger.info(f"L2→L1 Parent Accuracy: {report['summary']['l2_to_l1_accuracy']:.1%}")
        logger.info(f"L3→L2 Detailed Accuracy: {report['summary']['l3_to_l2_accuracy']:.1%}")
        logger.info(f"Hierarchy Quality: {report['summary']['hierarchy_quality']}")
    
    logger.info("\nAnalysis complete!")

if __name__ == "__main__":
    main()