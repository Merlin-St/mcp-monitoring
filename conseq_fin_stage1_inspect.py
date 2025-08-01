#!/usr/bin/env python3
"""
Financial MCP Server Stage 1 Filter - Inspect Task Definition

Uses Inspect framework to evaluate MCP servers for financial relevance
using LLM-based analysis with threat model mapping.

This file contains only the task definition for Inspect to run.
Use conseq_fin_stage1_dfprocessing.py to process the results.

Usage:
    inspect eval conseq_fin_stage1_inspect.py --model anthropic/claude-sonnet-4-20250514
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate, system_message


# Load environment variables
load_dotenv()



FINANCE_IDENTIFICATION_SYSTEM_PROMPT = """
ONLY RESPOND WITH JSON. NO MARKDOWN. NO TEXT OUTSIDE JSON.

You are a financial technology analyst identifying AI agent tools on model context protocol (MCP) servers - that interact
with financial systems. Follow this scoring process:

STEP 1: First, write analysis_notes
Analyze the MCP server and document your findings in the analysis_notes field. This should include:
- What the tool does and does not, any any uncertainty about this

STEP 2: Score is_finance_llm (0 or 1)
- 1: Tool interacts with financial systems, markets, or assets
- 0: Tool has no financial functionality

STEP 3: Identify asset_type
Valid values (can be multiple, separated by semicolon):
- "Banking" - Traditional bank accounts, transfers, payments
- "CreditCard" - Credit card operations, virtual cards
- "Crypto" - Cryptocurrency, blockchain, wallets, DeFi
- "Stocks" - Stock trading, investment platforms
- "Accounting" - Bookkeeping, financial reporting
- "Unclear" - When asset type cannot be determined

STEP 4: Set confidence level
- "H" (High): Comprehensive documentation, clear functionality
- "M" (Medium): Adequate documentation, some uncertainties
- "L" (Low): Limited/missing documentation, significant ambiguity

STEP 5: Determine level (0-5)
- 0: Not an MCP server - e.g. just a list that links to MCP servers
- 1: MONITORING - Read-only, no modification, just pull information
- 2: ANALYSIS - Get and analyse data, advise, does not modify external environment
- 3: META - Meta-server that connects to other servers that do execution
- 4: RESTRICTED EXECUTION - e.g. modify specified external environment, e.g. execute a transaction on specific blockchain
- 5: UNRESTRICTED EXECUTION - e.g. browser use with submitting info, full code execution for local & global edits

STEP 6: Score task capabilities (0 or 1 for each)
Capabilities to open a bank account:
- research_and_risk_assessment: Web search, rate aggregation, risk analysis,...
- documentation_gathering: File management, document scanning, PDF reading,...
- application_and_review: Form filling, browser automation,...
- identity_verification: Camera API, image upload, face recognition,...
- authorization_account_transactions: Email/SMS reading, OTP extraction, two-step authenticator...
- account_opening: Create wallets, operational accounts, subaccounts,...

Asset-specific transfer of money capabilities (only if these are specifically mentioned):
- transfer_bank_and_fund_bank_account: Banking API to transact to other bank, payment gateway,...
- transfer_credit_card: Use saved card info to pay,...
- transfer_paypal_stripe_payments: Payment provider APIs to execute transactions,...
- transfer_stock_invest: Place/modify/cancel stock orders,...
- transfer_crypto_and_stablecoin: Crypto and blockchain transactions,...

STEP 7: Identify sensitive data specifically required as an input (can be multiple, separated by semicolon. Copy from specific mentions in the documentation, do not guess):
- "API_edit_tokens" - API keys with full edit, e.g. transaction execution access
- "Credit_card_number"
- "cyrpto_wallet_address"
- "passport_number"
- "..." - Any other sensitive data that allows deep execution access that is required as an input - specify what it is

EXAMPLES:

Example 1 - Read-only banking monitor:
Input: asher-mcp
Description: Financial data aggregation tool
Tools:
- get_accounts: Retrieve list of all connected bank accounts
- get_account_balance: Get current balance for a specific account
- get_transactions: Retrieve transaction history for an account
- get_investment_holdings: View investment portfolio holdings
{
    "server": "asher-mcp",
    "analysis_notes": "Read-only access to own bank accounts via scraping",
    "is_finance_llm": 1,
    "asset_type": "Banking",
    "confidence": "H",
    "level": 1,
    "research_and_risk_assessment": 0,
    "documentation_gathering": 0,
    "application_and_review": 0,
    "identity_verification": 0,
    "authorization_account_transactions": 0,
    "account_opening": 0,
    "transfer_bank_and_fund_bank_account": 0,
    "transfer_credit_card": 0,
    "transfer_paypal_stripe_payments": 0,
    "transfer_stock_invest": 0,
    "transfer_crypto_and_stablecoin": 0,
    "sensitive_data_required": ""
}

Example 2 - Execution with limited transfer capabilities:
Input: base-mcp
Description: Blockchain interaction tool for Base network
Tools:
- get_balance: Check wallet balance
- get_transaction: Retrieve transaction details
- send_transaction: Send ETH or tokens
- deploy_contract: Deploy smart contracts
- interact_contract: Call contract functions
- estimate_gas: Calculate gas fees
Required inputs:
- private_key: Wallet private key
- rpc_endpoint: Base network RPC URL
{
    "server": "base-mcp",
    "analysis_notes": "Readme is insufficient (truncated)",
    "is_finance_llm": 1,
    "asset_type": "Crypto",
    "confidence": "L",
    "level": 4,
    "research_and_risk_assessment": 0,
    "documentation_gathering": 0,
    "application_and_review": 0,
    "identity_verification": 0,
    "authorization_account_transactions": 1,
    "account_opening": 0,
    "transfer_bank_and_fund_bank_account": 0,
    "transfer_credit_card": 0,
    "transfer_paypal_stripe_payments": 0,
    "transfer_stock_invest": 0,
    "transfer_crypto_and_stablecoin": 1,
    "sensitive_data_required": "private_key;rpc_endpoint"
}

Example 3 - Monitoring only crypto prices:
Input: binance-alpha-mcp
Description: Real-time cryptocurrency market monitoring
Tools:
- get_market_overview: Get overall market statistics
- get_trending_coins: View trending cryptocurrencies
- get_price_alerts: Monitor price movements
- get_volume_analysis: Analyze trading volumes across exchanges
{
    "server": "binance-alpha-mcp",
    "analysis_notes": "Monitoring only: monitoring of trades in the market (not specifically ones own activity)",
    "is_finance_llm": 1,
    "asset_type": "Crypto",
    "confidence": "H",
    "level": 1,
    "research_and_risk_assessment": 1,
    "documentation_gathering": 0,
    "application_and_review": 0,
    "identity_verification": 0,
    "authorization_account_transactions": 0,
    "account_opening": 0,
    "transfer_bank_and_fund_bank_account": 0,
    "transfer_credit_card": 0,
    "transfer_paypal_stripe_payments": 0,
    "transfer_stock_invest": 0,
    "transfer_crypto_and_stablecoin": 0,
    "sensitive_data_required": ""
}

Example 4 - Poor documentation:
Input: ai-agent-mcp-servers
Description: Collection of MCP servers for AI agents
Tools: [No tools listed in documentation]
{
    "server": "ai-agent-mcp-servers",
    "analysis_notes": "Almost no detail in the Readme",
    "is_finance_llm": 1,
    "asset_type": "Stocks",
    "confidence": "L",
    "level": 1,
    "research_and_risk_assessment": 0,
    "documentation_gathering": 0,
    "application_and_review": 0,
    "identity_verification": 0,
    "authorization_account_transactions": 0,
    "account_opening": 0,
    "transfer_bank_and_fund_bank_account": 0,
    "transfer_credit_card": 0,
    "transfer_paypal_stripe_payments": 0,
    "transfer_stock_invest": 0,
    "transfer_crypto_and_stablecoin": 0,
    "sensitive_data_required": ""
}

Example 5 - General computer use:
Input: DesktopCommanderMCP
Description: Execute python and control mouse and keyboard on local OS
Tools: Tools:
- execute_command: Execute arbitrary shell commands with timeout
- read_file: Read file contents with pagination / negative offset
- write_file: Write or append to files (line-limited)
- kill_process: Terminate a running process by PID

{
    "server": "DesktopCommanderMCP",
    "analysis_notes": "General-purpose MCP server for local automation: execute arbitrary terminal commands, manage processes, and perform full write operations on files. No tools mention financial data, payments, banking, crypto, or market interaction. Therefore it is not a finance-focused LLM",
    "is_finance_llm": 0,
    "asset_type": "Unclear",
    "confidence": "M",
    "level": 5,
    "research_and_risk_assessment": 0,
    "documentation_gathering": 1,
    "application_and_review": 0,
    "identity_verification": 0,
    "authorization_account_transactions": 0,
    "account_opening": 0,
    "transfer_bank_and_fund_bank_account": 0,
    "transfer_credit_card": 0,
    "transfer_paypal_stripe_payments": 0,
    "transfer_stock_invest": 0,
    "transfer_crypto_and_stablecoin": 0,
    "sensitive_data_required": ""
}

Output Format:
{
    "server": "string",
    "analysis_notes": "Brief analysis of the tool(s)",
    "is_finance_llm": 0|1,
    "asset_type": "Banking|CreditCard|Crypto|Stocks|Accounting|Unclear",
    "confidence": "H|M|L",
    "level": 0|1|2|3|4|5,
    "research_and_risk_assessment": 0|1,
    "documentation_gathering": 0|1,
    "application_and_review": 0|1,
    "identity_verification": 0|1,
    "authorization_account_transactions": 0|1,
    "account_opening": 0|1,
    "transfer_bank_and_fund_bank_account": 0|1,
    "transfer_credit_card": 0|1,
    "transfer_paypal_stripe_payments": 0|1,
    "transfer_stock_invest": 0|1,
    "transfer_crypto_and_stablecoin": 0|1,
    "sensitive_data_required": "API_edit_tokens|Credit_card_number|crypto_wallet_address|passport_number|...",
}

RESPOND ONLY WITH JSON.
""".strip()


@scorer(metrics=[accuracy()])
def finance_filter_scorer() -> Scorer:
    """
    Custom scorer for validating JSON structure and extracting results
    Tries to extract JSON from responses that might contain additional text
    """
    async def _scorer(state: TaskState, target: Target):
        completion = state.output.completion
        
        # Try to extract JSON from the completion text
        json_obj = None
        
        # First try: direct JSON parsing
        try:
            json_obj = json.loads(completion)
        except json.JSONDecodeError:
            # Second try: find JSON block in text
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, completion, re.DOTALL)
            
            for match in json_matches:
                try:
                    json_obj = json.loads(match)
                    break
                except json.JSONDecodeError:
                    continue
        
        if json_obj is None:
            # Third try: more aggressive JSON extraction
            try:
                # Look for content between first { and last }
                start = completion.find('{')
                end = completion.rfind('}')
                if start != -1 and end != -1 and end > start:
                    potential_json = completion[start:end+1]
                    json_obj = json.loads(potential_json)
            except json.JSONDecodeError:
                pass
        
        if json_obj is None:
            return Score(
                value=0,
                answer=completion,
                explanation="No valid JSON found in response"
            )
        
        # Validate all required fields
        required_fields = [
            "server",
            "analysis_notes", 
            "is_finance_llm",
            "asset_type",
            "confidence",
            "level",
            "research_and_risk_assessment",
            "documentation_gathering", 
            "application_and_review",
            "identity_verification",
            "authorization_account_transactions",
            "account_opening",
            "transfer_bank_and_fund_bank_account",
            "transfer_credit_card",
            "transfer_paypal_stripe_payments",
            "transfer_stock_invest",
            "transfer_crypto_and_stablecoin",
            "sensitive_data_required"
        ]
        
        missing_fields = [field for field in required_fields if field not in json_obj]
        
        if missing_fields:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Missing required fields: {missing_fields}"
            )
        
        # Validate binary fields (should be 0 or 1)
        binary_fields = [
            "is_finance_llm",
            "research_and_risk_assessment",
            "documentation_gathering", 
            "application_and_review",
            "identity_verification",
            "authorization_account_transactions",
            "account_opening",
            "transfer_bank_and_fund_bank_account",
            "transfer_credit_card",
            "transfer_paypal_stripe_payments",
            "transfer_stock_invest",
            "transfer_crypto_and_stablecoin"
        ]
        
        for field in binary_fields:
            if json_obj[field] not in [0, 1]:
                return Score(
                    value=0,
                    answer=completion,
                    explanation=f"Invalid {field} value: {json_obj[field]} (expected 0 or 1)"
                )
        
        # Validate level field (should be 0-5)
        if json_obj["level"] not in [0, 1, 2, 3, 4, 5]:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Invalid level value: {json_obj['level']} (expected 0-5)"
            )
        
        # Validate confidence field (should be H, M, or L)
        if json_obj["confidence"] not in ["H", "M", "L"]:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Invalid confidence value: {json_obj['confidence']} (expected H, M, or L)"
            )
        
        return Score(
            value=1,
            answer=completion,
            explanation="Valid JSON with required fields extracted"
        )
    
    return _scorer

def count_dataset_size(dataset_file):
    """Count the number of samples in the dataset file"""
    if not Path(dataset_file).exists():
        return 0
    
    with open(dataset_file, 'r') as f:
        count = sum(1 for _ in f)
    
    return count

@task
def finance_identification_task():
    """
    Inspect task for identifying finance-related MCP servers using system_message solver
    """
    dataset_file = "conseq_fin_stage1_input.jsonl"
    
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found")
    
    # Count samples in dataset to set appropriate message limit
    dataset_size = count_dataset_size(dataset_file)
    dynamic_message_limit = dataset_size + 10  # Add buffer for safety
    
    
    
    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message(FINANCE_IDENTIFICATION_SYSTEM_PROMPT),
            generate()
        ],
        scorer=[finance_filter_scorer()],
        message_limit=dynamic_message_limit
    )

# API key handling is managed automatically by AISI environment variables:
# - INSPECT_API_KEY_OVERRIDE=aisitools.api_key.override_api_key
# - ANTHROPIC_API_KEY (handled automatically by inspect framework)