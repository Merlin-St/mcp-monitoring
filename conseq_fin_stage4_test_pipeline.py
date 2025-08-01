#!/usr/bin/env python3
"""
Test script to validate the O*NET classification pipeline setup
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hierarchy():
    """Test hierarchy file exists and is valid"""
    logger.info("Testing hierarchy...")
    hierarchy_file = Path("conseq_fin_stage4_hierarchy.json")
    
    if not hierarchy_file.exists():
        logger.error("❌ Hierarchy file not found. Run: python conseq_fin_stage4_embed_levels.py")
        return False
    
    with open(hierarchy_file, 'r') as f:
        hierarchy = json.load(f)
    
    # Validate structure
    required_keys = ['metadata', 'top_level', 'middle_level', 'task_lookup']
    for key in required_keys:
        if key not in hierarchy:
            logger.error(f"❌ Missing key in hierarchy: {key}")
            return False
    
    logger.info(f"✅ Hierarchy loaded: {len(hierarchy['task_lookup'])} tasks mapped")
    return True

def test_tool_data():
    """Test tool data preparation"""
    logger.info("Testing tool data...")
    
    input_file = Path("conseq_fin_stage4_input.jsonl")
    if not input_file.exists():
        logger.error("❌ Input file not found. Run: python conseq_fin_stage4_data_prep.py")
        return False
    
    # Count samples
    with open(input_file, 'r') as f:
        sample_count = sum(1 for _ in f)
    
    logger.info(f"✅ Tool data ready: {sample_count} samples")
    
    # Check sample structure
    with open(input_file, 'r') as f:
        sample = json.loads(f.readline())
    
    required_fields = ['input', 'target', 'id', 'metadata']
    for field in required_fields:
        if field not in sample:
            logger.error(f"❌ Missing field in sample: {field}")
            return False
    
    # Parse input data
    input_data = json.loads(sample['input'])
    tool_fields = ['tool_name', 'tool_description', 'server_name', 'server_description']
    for field in tool_fields:
        if field not in input_data:
            logger.error(f"❌ Missing field in tool data: {field}")
            return False
    
    logger.info(f"✅ Sample tool: {input_data['tool_name']} from {input_data['server_name']}")
    return True

def test_onet_tools():
    """Test O*NET tools file"""
    logger.info("Testing O*NET tools...")
    
    tools_file = Path("conseq_fin_stage4_onet_toolsused.csv")
    if not tools_file.exists():
        logger.error("❌ O*NET tools file not found")
        return False
    
    import pandas as pd
    tools_df = pd.read_csv(tools_file)
    unique_tools = tools_df['Example'].nunique()
    
    logger.info(f"✅ O*NET tools loaded: {unique_tools} unique tools")
    return True

def test_scripts():
    """Test all scripts are present"""
    logger.info("Testing scripts...")
    
    scripts = [
        "conseq_fin_stage4_embed_levels.py",
        "conseq_fin_stage4_data_prep.py", 
        "conseq_fin_stage4_inspect.py",
        "conseq_fin_stage4_dfprocessing.py",
        "conseq_fin_stage4_analysis.py"
    ]
    
    all_present = True
    for script in scripts:
        if Path(script).exists():
            logger.info(f"✅ {script}")
        else:
            logger.error(f"❌ {script} not found")
            all_present = False
    
    return all_present

def test_dependencies():
    """Test required dependencies"""
    logger.info("Testing dependencies...")
    
    try:
        import sentence_transformers
        logger.info("✅ sentence-transformers")
    except ImportError:
        logger.error("❌ sentence-transformers not installed")
        return False
    
    try:
        import sklearn
        logger.info("✅ scikit-learn")
    except ImportError:
        logger.error("❌ scikit-learn not installed")
        return False
    
    try:
        import inspect_ai
        logger.info("✅ inspect_ai")
    except ImportError:
        logger.error("❌ inspect_ai not installed")
        return False
    
    try:
        import matplotlib
        import seaborn
        logger.info("✅ visualization libraries")
    except ImportError:
        logger.error("❌ matplotlib/seaborn not installed")
        return False
    
    return True

def main():
    logger.info("🔍 O*NET Classification Pipeline Test")
    logger.info("="*50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Scripts", test_scripts),
        ("Hierarchy", test_hierarchy),
        ("Tool Data", test_tool_data),
        ("O*NET Tools", test_onet_tools)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        if not test_func():
            all_passed = False
    
    logger.info("\n" + "="*50)
    if all_passed:
        logger.info("✅ All tests passed! Pipeline is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Run full hierarchy: python conseq_fin_stage4_embed_levels.py")
        logger.info("2. Prepare all tools: python conseq_fin_stage4_data_prep.py --all")
        logger.info("3. Run classification: inspect eval conseq_fin_stage4_inspect.py --model anthropic/claude-sonnet-4-20250514")
        logger.info("4. Process results: python conseq_fin_stage4_dfprocessing.py")
        logger.info("5. Generate analysis: python conseq_fin_stage4_analysis.py")
    else:
        logger.error("❌ Some tests failed. Please fix issues before running pipeline.")

if __name__ == "__main__":
    main()