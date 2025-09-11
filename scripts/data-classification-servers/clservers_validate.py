#!/usr/bin/env python3
"""
Finance Consequentiality CLServers Validation Script

Compares human-labeled CSV (2025-07-labelling-finance-mcps.csv) against 
LLM-labeled CSV (server_classified.csv) to evaluate accuracy and provide
comprehensive validation metrics for each overlapping field.

Inspired by the stage1 inspect validation methodology.
"""

import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/clservers_validate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load human-labeled and LLM-labeled datasets."""
    logger.info("Loading datasets...")
    
    # Load human-labeled data
    human_df = pd.read_csv("data/external-cl/clservers_validate_labelled.csv")
    logger.info(f"Human-labeled data: {len(human_df)} rows, {len(human_df.columns)} columns")
    
    # Load LLM-labeled data
    llm_df = pd.read_csv("data/final/clservers_classified.csv")
    logger.info(f"LLM-labeled data: {len(llm_df)} rows, {len(llm_df.columns)} columns")
    
    return human_df, llm_df


def align_datasets(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align datasets by server_id and identify overlapping records."""
    logger.info("Aligning datasets by server_id...")
    
    # Clean server_id columns
    human_df['server_id'] = human_df['server_id'].astype(str).str.strip()
    llm_df['server_id'] = llm_df['server_id'].astype(str).str.strip()
    
    # Find common server_ids
    human_ids = set(human_df['server_id'].unique())
    llm_ids = set(llm_df['server_id'].unique())
    common_ids = human_ids.intersection(llm_ids)
    
    logger.info(f"Human-labeled unique IDs: {len(human_ids)}")
    logger.info(f"LLM-labeled unique IDs: {len(llm_ids)}")
    logger.info(f"Common IDs: {len(common_ids)}")
    logger.info(f"Only in human: {len(human_ids - llm_ids)}")
    logger.info(f"Only in LLM: {len(llm_ids - human_ids)}")
    
    # Filter to common records
    human_aligned = human_df[human_df['server_id'].isin(common_ids)].copy()
    llm_aligned = llm_df[llm_df['server_id'].isin(common_ids)].copy()
    
    # Sort by server_id for consistent alignment
    human_aligned = human_aligned.sort_values('server_id').reset_index(drop=True)
    llm_aligned = llm_aligned.sort_values('server_id').reset_index(drop=True)
    
    return human_aligned, llm_aligned


def identify_comparable_fields() -> Dict[str, str]:
    """Map human column names to LLM column names for comparison."""
    field_mapping = {
        # Binary finance classification
        'Is_finance': 'is_finance_llm',
        
        # Asset type classification
        'Asset_type': 'asset_type',
        
        # Consequentiality level
        'Level ': 'level',  # Note: human has trailing space
        
        # Capability fields (binary indicators)
        'research_and_risk_assessment': 'research_and_risk_assessment',
        'documentation_gathering': 'documentation_gathering',
        'application_and_review': 'application_and_review',
        'identity_verification': 'identity_verification',
        'account_opening': 'account_opening',
        'account_activation_and_transaction_authorization': 'authorization_account_transactions',
        'transfer_bank_and_fund_bank_ account': 'transfer_bank_and_fund_bank_account',
        'transfer_credit_card': 'transfer_credit_card',
        'transfer_paypal-stripe-_payments': 'transfer_paypal_stripe_payments',
        'transfer_stock_invest': 'transfer_stock_invest',
        'transfer_crypto_and_stablecoin': 'transfer_crypto_and_stablecoin',
        'Capability: sensitive_keys_required': 'sensitive_data_required',
    }
    
    return field_mapping


def calculate_binary_metrics(human_values: pd.Series, llm_values: pd.Series, 
                           field_name: str) -> Dict[str, Any]:
    """Calculate metrics for binary classification fields."""
    # Handle missing/NaN values
    valid_mask = ~(human_values.isna() | llm_values.isna())
    h_vals = human_values[valid_mask]
    l_vals = llm_values[valid_mask]
    
    if len(h_vals) == 0:
        return {
            'total_samples': 0,
            'valid_samples': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'confusion_matrix': [[0, 0], [0, 0]]
        }
    
    # Convert to binary (handle different representations)
    def normalize_binary(vals):
        # Convert to string and normalize
        str_vals = vals.astype(str).str.lower().str.strip()
        # Map common representations to binary
        binary_map = {'1': 1, '1.0': 1, 'true': 1, 'yes': 1, 'y': 1,
                     '0': 0, '0.0': 0, 'false': 0, 'no': 0, 'n': 0, 
                     'nan': 0, 'none': 0, '': 0}
        return str_vals.map(binary_map).fillna(0).astype(int)
    
    h_binary = normalize_binary(h_vals)
    l_binary = normalize_binary(l_vals)
    
    # Calculate confusion matrix
    tp = ((h_binary == 1) & (l_binary == 1)).sum()
    tn = ((h_binary == 0) & (l_binary == 0)).sum()
    fp = ((h_binary == 0) & (l_binary == 1)).sum()
    fn = ((h_binary == 1) & (l_binary == 0)).sum()
    
    confusion_matrix = [[tn, fp], [fn, tp]]
    
    # Calculate metrics
    accuracy = (tp + tn) / len(h_binary) if len(h_binary) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_samples': len(human_values),
        'valid_samples': len(h_binary),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def calculate_categorical_metrics(human_values: pd.Series, llm_values: pd.Series,
                                field_name: str) -> Dict[str, Any]:
    """Calculate metrics for categorical classification fields."""
    # Handle missing/NaN values
    valid_mask = ~(human_values.isna() | llm_values.isna())
    h_vals = human_values[valid_mask].astype(str).str.strip()
    l_vals = llm_values[valid_mask].astype(str).str.strip()
    
    if len(h_vals) == 0:
        return {
            'total_samples': 0,
            'valid_samples': 0,
            'accuracy': 0,
            'unique_human_values': [],
            'unique_llm_values': [],
            'value_distribution': {}
        }
    
    # Get unique values
    unique_human = sorted(h_vals.unique())
    unique_llm = sorted(l_vals.unique())
    all_values = sorted(set(unique_human + unique_llm))
    
    # Calculate accuracy
    accuracy = (h_vals == l_vals).mean()
    
    # Value distribution analysis
    value_distribution = {}
    for val in all_values:
        human_count = (h_vals == val).sum()
        llm_count = (l_vals == val).sum()
        value_distribution[val] = {
            'human_count': int(human_count),
            'llm_count': int(llm_count)
        }
    
    return {
        'total_samples': len(human_values),
        'valid_samples': len(h_vals),
        'accuracy': accuracy,
        'unique_human_values': unique_human,
        'unique_llm_values': unique_llm,
        'value_distribution': value_distribution
    }


def calculate_level_metrics(human_values: pd.Series, llm_values: pd.Series) -> Dict[str, Any]:
    """Calculate specialized metrics for consequentiality level comparison."""
    # Handle missing/NaN values and convert to numeric
    valid_mask = ~(human_values.isna() | llm_values.isna())
    h_vals = pd.to_numeric(human_values[valid_mask], errors='coerce').reset_index(drop=True)
    l_vals = pd.to_numeric(llm_values[valid_mask], errors='coerce').reset_index(drop=True)
    
    # Remove any remaining NaN values after numeric conversion
    valid_mask2 = ~(h_vals.isna() | l_vals.isna())
    h_vals = h_vals[valid_mask2].reset_index(drop=True)
    l_vals = l_vals[valid_mask2].reset_index(drop=True)
    
    if len(h_vals) == 0:
        return {
            'total_samples': 0,
            'valid_samples': 0,
            'accuracy': 0,
            'off_by_one_accuracy': 0,
            'mean_absolute_error': 0,
            'confusion_matrix': []
        }
    
    # Ensure values are in expected range (1-5) and maintain alignment
    valid_range_mask = (h_vals >= 1) & (h_vals <= 5) & (l_vals >= 1) & (l_vals <= 5)
    h_vals = h_vals[valid_range_mask].reset_index(drop=True)
    l_vals = l_vals[valid_range_mask].reset_index(drop=True)
    
    if len(h_vals) == 0:
        return {
            'total_samples': 0,
            'valid_samples': 0,
            'accuracy': 0,
            'off_by_one_accuracy': 0,
            'mean_absolute_error': 0,
            'confusion_matrix': []
        }
    
    # Convert to numpy arrays for easier computation
    h_array = h_vals.values
    l_array = l_vals.values
    
    # Calculate metrics
    exact_match = np.sum(h_array == l_array)
    off_by_one = np.sum(np.abs(h_array - l_array) <= 1)
    mae = np.mean(np.abs(h_array - l_array))
    
    accuracy = exact_match / len(h_array)
    off_by_one_accuracy = off_by_one / len(h_array)
    
    # Build confusion matrix (5x5 for levels 1-5)
    confusion_matrix = [[0 for _ in range(5)] for _ in range(5)]
    for h, llm_score in zip(h_array, l_array):
        confusion_matrix[int(h)-1][int(llm_score)-1] += 1
    
    # Analyze specific disagreement patterns
    disagreements = []
    for i, (h, llm_score) in enumerate(zip(h_array, l_array)):
        if h != llm_score:
            disagreements.append({
                'index': i,
                'human_level': int(h),
                'llm_level': int(llm_score),
                'difference': int(llm_score) - int(h)
            })
    
    # Calculate level distribution
    human_distribution = {level: int(np.sum(h_array == level)) for level in range(1, 6)}
    llm_distribution = {level: int(np.sum(l_array == level)) for level in range(1, 6)}
    
    return {
        'total_samples': len(human_values),
        'valid_samples': len(h_array),
        'accuracy': accuracy,
        'off_by_one_accuracy': off_by_one_accuracy,
        'mean_absolute_error': mae,
        'confusion_matrix': confusion_matrix,
        'disagreements': disagreements,
        'human_distribution': human_distribution,
        'llm_distribution': llm_distribution,
        'total_disagreements': len(disagreements)
    }


def perform_validation_analysis(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive validation analysis across all comparable fields."""
    logger.info("Starting validation analysis...")
    
    field_mapping = identify_comparable_fields()
    results = {}
    
    for human_field, llm_field in field_mapping.items():
        logger.info(f"Analyzing field: {human_field} -> {llm_field}")
        
        if human_field not in human_df.columns:
            logger.warning(f"Human field '{human_field}' not found in dataset")
            continue
        if llm_field not in llm_df.columns:
            logger.warning(f"LLM field '{llm_field}' not found in dataset")
            continue
        
        human_values = human_df[human_field]
        llm_values = llm_df[llm_field]
        
        # Special handling for consequentiality level
        if human_field.strip().lower() in ['level', 'level ']:
            field_results = calculate_level_metrics(human_values, llm_values)
        # Check if field appears to be binary
        elif human_field in ['Is_finance'] or 'transfer_' in human_field or 'research_' in human_field or 'Capability:' in human_field:
            field_results = calculate_binary_metrics(human_values, llm_values, human_field)
        else:
            field_results = calculate_categorical_metrics(human_values, llm_values, human_field)
        
        field_results['human_field'] = human_field
        field_results['llm_field'] = llm_field
        results[human_field] = field_results
    
    return results


def generate_summary_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics across all validated fields."""
    logger.info("Generating summary report...")
    
    # Overall statistics
    total_fields = len(results)
    binary_fields = []
    categorical_fields = []
    level_fields = []
    
    accuracies = []
    f1_scores = []
    
    for field, metrics in results.items():
        if 'confusion_matrix' in metrics and len(metrics['confusion_matrix']) == 2:
            binary_fields.append(field)
            accuracies.append(metrics['accuracy'])
            if 'f1' in metrics:
                f1_scores.append(metrics['f1'])
        elif 'confusion_matrix' in metrics and len(metrics['confusion_matrix']) == 5:
            level_fields.append(field)
            accuracies.append(metrics['accuracy'])
        else:
            categorical_fields.append(field)
            accuracies.append(metrics['accuracy'])
    
    # Calculate summary metrics
    mean_accuracy = np.mean(accuracies) if accuracies else 0
    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    
    summary = {
        'total_fields_analyzed': total_fields,
        'binary_fields_count': len(binary_fields),
        'categorical_fields_count': len(categorical_fields),
        'level_fields_count': len(level_fields),
        'mean_accuracy': mean_accuracy,
        'mean_f1_score': mean_f1,
        'binary_fields': binary_fields,
        'categorical_fields': categorical_fields,
        'level_fields': level_fields,
        'accuracy_distribution': {
            'min': min(accuracies) if accuracies else 0,
            'max': max(accuracies) if accuracies else 0,
            'mean': mean_accuracy,
            'std': np.std(accuracies) if accuracies else 0
        }
    }
    
    return summary


def print_detailed_report(results: Dict[str, Any], summary: Dict[str, Any]):
    """Print comprehensive validation report."""
    logger.info("=== FINANCE CONSEQUENTIALITY VALIDATION REPORT ===")
    logger.info(f"Total fields analyzed: {summary['total_fields_analyzed']}")
    logger.info(f"Binary fields: {summary['binary_fields_count']}")
    logger.info(f"Categorical fields: {summary['categorical_fields_count']}")
    logger.info(f"Level fields: {summary['level_fields_count']}")
    logger.info(f"Mean accuracy across all fields: {summary['mean_accuracy']:.4f}")
    logger.info(f"Mean F1 score (binary fields): {summary['mean_f1_score']:.4f}")
    
    logger.info("\n=== DETAILED FIELD ANALYSIS ===")
    
    for field, metrics in results.items():
        logger.info(f"\n--- {field} -> {metrics['llm_field']} ---")
        logger.info(f"Valid samples: {metrics['valid_samples']}/{metrics['total_samples']}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        
        if 'f1' in metrics:
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1']:.4f}")
            logger.info(f"Confusion Matrix: {metrics['confusion_matrix']}")
        
        if 'off_by_one_accuracy' in metrics:
            logger.info(f"Off-by-one accuracy: {metrics['off_by_one_accuracy']:.4f}")
            logger.info(f"Mean absolute error: {metrics['mean_absolute_error']:.4f}")
            logger.info(f"Total disagreements: {metrics.get('total_disagreements', 'N/A')}")
            
            # Detailed level analysis
            if 'human_distribution' in metrics and 'llm_distribution' in metrics:
                logger.info("Level distribution comparison:")
                for level in range(1, 6):
                    h_count = metrics['human_distribution'].get(level, 0)
                    l_count = metrics['llm_distribution'].get(level, 0)
                    logger.info(f"  Level {level}: Human={h_count}, LLM={l_count}")
            
            # Disagreement pattern analysis
            if 'disagreements' in metrics and metrics['disagreements']:
                disagreements = metrics['disagreements']
                logger.info("Disagreement patterns:")
                
                # Count by difference magnitude
                diff_counts = {}
                for d in disagreements:
                    diff = d['difference']
                    diff_counts[diff] = diff_counts.get(diff, 0) + 1
                
                for diff in sorted(diff_counts.keys()):
                    count = diff_counts[diff]
                    direction = "over-estimated" if diff > 0 else "under-estimated"
                    logger.info(f"  LLM {direction} by {abs(diff)}: {count} cases")
            
            # Confusion matrix analysis for levels
            if 'confusion_matrix' in metrics:
                logger.info("Confusion matrix (rows=Human, cols=LLM):")
                cm = metrics['confusion_matrix']
                logger.info("    " + " ".join([f"L{i+1:2}" for i in range(5)]))
                for i, row in enumerate(cm):
                    logger.info(f"L{i+1}: " + " ".join([f"{val:2}" for val in row]))
        
        if 'value_distribution' in metrics:
            logger.info("Value distribution:")
            for val, counts in metrics['value_distribution'].items():
                logger.info(f"  '{val}': Human={counts['human_count']}, LLM={counts['llm_count']}")


def save_results(results: Dict[str, Any], summary: Dict[str, Any], 
                 human_df: pd.DataFrame, llm_df: pd.DataFrame):
    """Save validation results to files."""
    logger.info("Saving validation results...")
    
    # Prepare detailed results for JSON serialization
    json_results = {}
    for field, metrics in results.items():
        # Convert numpy types to native Python types
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                json_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, list):
                # Handle nested lists/matrices with numpy types
                json_metrics[key] = [[int(x) if isinstance(x, (np.integer, np.int64)) else x for x in row] if isinstance(row, list) else row for row in value]
            else:
                json_metrics[key] = value
        json_results[field] = json_metrics
    
    # Convert summary numpy types
    json_summary = {}
    for key, value in summary.items():
        if isinstance(value, (np.integer, np.floating)):
            json_summary[key] = float(value)
        elif isinstance(value, dict):
            converted_dict = {}
            for k, v in value.items():
                if isinstance(v, (np.integer, np.floating)):
                    converted_dict[k] = float(v)
                else:
                    converted_dict[k] = v
            json_summary[key] = converted_dict
        else:
            json_summary[key] = value
    
    # Save comprehensive results
    output = {
        'summary': json_summary,
        'field_results': json_results,
        'dataset_info': {
            'human_records': len(human_df),
            'llm_records': len(llm_df),
            'common_records': json_summary.get('total_fields_analyzed', 0)
        }
    }
    
    with open('output-validation/cl-validation/clservers_validation.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info("Validation results saved to output-validation/cl-validation/clservers_validation.json")


def main():
    """Main validation workflow."""
    logger.info("Starting Finance Consequentiality CLServers Validation")
    
    try:
        # Load and align datasets
        human_df, llm_df = load_data()
        human_aligned, llm_aligned = align_datasets(human_df, llm_df)
        
        if len(human_aligned) == 0:
            logger.error("No common records found between datasets")
            return
        
        # Perform validation analysis
        results = perform_validation_analysis(human_aligned, llm_aligned)
        
        if not results:
            logger.error("No comparable fields found for analysis")
            return
        
        # Generate summary and report
        summary = generate_summary_report(results)
        print_detailed_report(results, summary)
        save_results(results, summary, human_aligned, llm_aligned)
        
        logger.info("Validation analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()