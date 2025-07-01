#!/usr/bin/env python3
"""
Financial MCP Server Results Merger

Combines results from Stage 1 (finance identification) and Stage 2 (consequentiality assessment)
with original server data to create a comprehensive final analysis.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conseq_fin_results_merger.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_original_servers() -> Dict[str, Dict[str, Any]]:
    """Load original server data from the sample file"""
    servers_file = "conseq_fin_servers_sample.json"
    
    if not Path(servers_file).exists():
        logger.error(f"Original servers file {servers_file} not found")
        raise FileNotFoundError(f"Original servers file not found")
    
    with open(servers_file, 'r', encoding='utf-8') as f:
        servers_list = json.load(f)
    
    # Convert to dict for easier lookup
    servers_dict = {server['id']: server for server in servers_list}
    logger.info(f"Loaded {len(servers_dict)} original servers")
    
    return servers_dict

def load_stage1_results() -> Dict[str, Dict[str, Any]]:
    """Load Stage 1 results"""
    stage1_file = "conseq_fin_stage1_results.json"
    
    if not Path(stage1_file).exists():
        logger.error(f"Stage 1 results file {stage1_file} not found")
        raise FileNotFoundError(f"Stage 1 results file not found")
    
    with open(stage1_file, 'r', encoding='utf-8') as f:
        stage1_data = json.load(f)
    
    # Convert results to dict for easier lookup
    results_dict = {}
    for result in stage1_data["results"]:
        server_id = result["sample_id"]
        results_dict[server_id] = result
    
    logger.info(f"Loaded Stage 1 results for {len(results_dict)} servers")
    return results_dict, stage1_data["summary"]

def load_stage2_results() -> Dict[str, Dict[str, Any]]:
    """Load Stage 2 results if available"""
    stage2_file = "conseq_fin_stage2_results.json"
    
    if not Path(stage2_file).exists():
        logger.warning(f"Stage 2 results file {stage2_file} not found. Only Stage 1 results will be included.")
        return {}, {}
    
    with open(stage2_file, 'r', encoding='utf-8') as f:
        stage2_data = json.load(f)
    
    # Convert results to dict for easier lookup
    results_dict = {}
    for result in stage2_data["results"]:
        server_id = result["sample_id"]
        results_dict[server_id] = result
    
    logger.info(f"Loaded Stage 2 results for {len(results_dict)} servers")
    return results_dict, stage2_data["summary"]

def classify_risk_category(consequentiality_level: Optional[int]) -> str:
    """Classify risk category based on consequentiality level"""
    if consequentiality_level is None:
        return "UNKNOWN"
    elif consequentiality_level == 1:
        return "LOW"
    elif consequentiality_level == 2:
        return "LOW-MEDIUM"
    elif consequentiality_level == 3:
        return "MEDIUM"
    elif consequentiality_level == 4:
        return "HIGH"
    elif consequentiality_level == 5:
        return "CRITICAL"
    else:
        return "UNKNOWN"

def extract_threat_models(stage1_result: Dict[str, Any]) -> List[str]:
    """Extract threat model identifiers from Stage 1 results"""
    threat_models = []
    
    if (stage1_result.get("parsed_output") and 
        stage1_result["parsed_output"].get("threat_models")):
        
        for tm in stage1_result["parsed_output"]["threat_models"]:
            if isinstance(tm, dict) and "model" in tm:
                threat_models.append(tm["model"])
    
    return threat_models

def merge_results(original_servers: Dict[str, Dict[str, Any]], 
                 stage1_results: Dict[str, Dict[str, Any]],
                 stage2_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge all results into comprehensive server analysis"""
    
    merged_servers = []
    
    for server_id, server_data in original_servers.items():
        merged_server = {
            "server_data": server_data,
            "stage1_results": None,
            "stage2_results": None,
            "final_classification": {
                "is_finance_llm": "no",
                "threat_models": [],
                "consequentiality_level": None,
                "risk_category": "NOT_FINANCE"
            }
        }
        
        # Add Stage 1 results if available
        if server_id in stage1_results:
            stage1_result = stage1_results[server_id]
            merged_server["stage1_results"] = stage1_result
            
            # Extract classification from Stage 1
            if stage1_result.get("parsed_output"):
                stage1_output = stage1_result["parsed_output"]
                is_finance = stage1_output.get("is_finance_llm", "no")
                
                merged_server["final_classification"]["is_finance_llm"] = is_finance
                
                if is_finance == "yes":
                    merged_server["final_classification"]["threat_models"] = extract_threat_models(stage1_result)
                    merged_server["final_classification"]["risk_category"] = "FINANCE_IDENTIFIED"
        
        # Add Stage 2 results if available
        if server_id in stage2_results:
            stage2_result = stage2_results[server_id]
            merged_server["stage2_results"] = stage2_result
            
            # Extract consequentiality assessment from Stage 2
            if stage2_result.get("parsed_output"):
                stage2_output = stage2_result["parsed_output"]
                consequentiality_level = stage2_output.get("consequentiality_level")
                
                if isinstance(consequentiality_level, int):
                    merged_server["final_classification"]["consequentiality_level"] = consequentiality_level
                    merged_server["final_classification"]["risk_category"] = classify_risk_category(consequentiality_level)
        
        merged_servers.append(merged_server)
    
    return merged_servers

def generate_comprehensive_summary(merged_servers: List[Dict[str, Any]], 
                                 stage1_summary: Dict[str, Any],
                                 stage2_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive summary statistics"""
    
    # Count classifications
    finance_counts = {"yes": 0, "no": 0, "unclear": 0}
    threat_model_counts = {"TM1": 0, "TM2": 0, "TM3": 0}
    consequentiality_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    risk_category_counts = {}
    
    # Data source distribution
    data_source_counts = {}
    tools_distribution = {"no_tools": 0, "1-5_tools": 0, "6-10_tools": 0, "11+_tools": 0}
    
    for server in merged_servers:
        # Finance classification
        is_finance = server["final_classification"]["is_finance_llm"]
        finance_counts[is_finance] = finance_counts.get(is_finance, 0) + 1
        
        # Threat models
        for tm in server["final_classification"]["threat_models"]:
            threat_model_counts[tm] = threat_model_counts.get(tm, 0) + 1
        
        # Consequentiality levels
        level = server["final_classification"]["consequentiality_level"]
        if level is not None:
            consequentiality_counts[level] = consequentiality_counts.get(level, 0) + 1
        
        # Risk categories
        risk_cat = server["final_classification"]["risk_category"]
        risk_category_counts[risk_cat] = risk_category_counts.get(risk_cat, 0) + 1
        
        # Data sources
        source = server["server_data"].get("primary_source", "unknown")
        data_source_counts[source] = data_source_counts.get(source, 0) + 1
        
        # Tools distribution
        tools_count = server["server_data"].get("tools_count", 0)
        if tools_count == 0:
            tools_distribution["no_tools"] += 1
        elif tools_count <= 5:
            tools_distribution["1-5_tools"] += 1
        elif tools_count <= 10:
            tools_distribution["6-10_tools"] += 1
        else:
            tools_distribution["11+_tools"] += 1
    
    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "total_servers": len(merged_servers),
        "finance_classification": finance_counts,
        "threat_model_distribution": threat_model_counts,
        "consequentiality_distribution": consequentiality_counts,
        "risk_category_distribution": risk_category_counts,
        "data_source_distribution": data_source_counts,
        "tools_distribution": tools_distribution,
        "stage1_summary": stage1_summary,
        "stage2_summary": stage2_summary,
        "high_risk_servers": len([s for s in merged_servers 
                                if s["final_classification"]["risk_category"] in ["HIGH", "CRITICAL"]]),
        "average_consequentiality": (
            sum(level * count for level, count in consequentiality_counts.items()) / 
            sum(consequentiality_counts.values())
        ) if sum(consequentiality_counts.values()) > 0 else None
    }
    
    return summary

def main():
    """Main function to merge all results"""
    logger.info("Starting results merger")
    
    try:
        # Load all data sources
        logger.info("Loading original server data...")
        original_servers = load_original_servers()
        
        logger.info("Loading Stage 1 results...")
        stage1_results, stage1_summary = load_stage1_results()
        
        logger.info("Loading Stage 2 results...")
        stage2_results, stage2_summary = load_stage2_results()
        
        # Merge results
        logger.info("Merging results...")
        merged_servers = merge_results(original_servers, stage1_results, stage2_results)
        
        # Generate summary
        logger.info("Generating comprehensive summary...")
        summary = generate_comprehensive_summary(merged_servers, stage1_summary, stage2_summary)
        
        # Create final output
        final_output = {
            "summary": summary,
            "servers": merged_servers
        }
        
        # Save results
        output_file = "conseq_fin_final_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Final results saved to {output_file}")
        
        # Log summary statistics
        logger.info("=== FINAL ANALYSIS SUMMARY ===")
        logger.info(f"Total servers analyzed: {summary['total_servers']}")
        logger.info(f"Finance-related servers: {summary['finance_classification']['yes']}")
        logger.info(f"High-risk servers (HIGH/CRITICAL): {summary['high_risk_servers']}")
        
        if summary['average_consequentiality']:
            logger.info(f"Average consequentiality level: {summary['average_consequentiality']:.2f}")
        
        logger.info(f"Risk category distribution: {summary['risk_category_distribution']}")
        logger.info(f"Threat model distribution: {summary['threat_model_distribution']}")
        
        # Save summary only file for quick reference
        summary_file = "conseq_fin_summary_only.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to {summary_file}")
        logger.info("Results merger completed successfully!")
        
    except Exception as e:
        logger.error(f"Results merger failed: {e}")
        raise

if __name__ == "__main__":
    main()