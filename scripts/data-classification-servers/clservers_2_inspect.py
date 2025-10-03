#!/usr/bin/env python3
"""
CLServers Step 2: Financial MCP Server Filter - Inspect Task Definition

Uses Inspect framework to evaluate MCP servers for financial relevance
using LLM-based analysis with threat model mapping.

This file contains only the task definition for Inspect to run.
Use clservers_3_dfprocessing.py to process the results.

Usage:
    inspect eval clservers_2_inspect.py --model anthropic/claude-sonnet-4-5-20250929 --temperature 0
    inspect eval clservers_2_inspect.py@naics_classification_task --model anthropic/claude-sonnet-4-5-20250929 --temperature 0
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate, system_message

from naics_3digit_data import format_naics_for_prompt

# Load environment variables
load_dotenv()



FINANCE_IDENTIFICATION_SYSTEM_PROMPT = """
ONLY RESPOND WITH JSON. NO MARKDOWN. NO TEXT OUTSIDE JSON.

You are a technology analyst identifying AI agent tools on model context protocol (MCP) servers. Follow this scoring process:

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

STEP 7: DESCRIBE action space:
- action_space_description: Describe what action space/environment this server operates in and its focus
  * Example 1: "Pre-defined bitcoin wallet operations, in coinbase environment via API"
  * Example 2: "General-purpose file system access across any directory and file type"

STEP 8: CLASSIFY generality (based on action space):
- generality_industry: 1 if cross-industry (usable across many sectors), 0 if industry-specific
  * Examples of 1 (cross-industry): desktop commander, file management, email client, calendar tools, note-taking
  * Examples of 0 (industry-specific): crypto transaction tools, payment platforms, medical records systems, legal document processors

- generality_environment: 1 if open/untrusted environment, 0 if trusted/pre-specified environment
  * Examples of 1 (open/untrusted): computer use, browser automation, web scraping, generic file system access
  * Examples of 0 (trusted): specific API (Stripe, GitHub, Slack), internal database, pre-configured service endpoint

STEP 9: Identify sensitive data specifically required as an input (can be multiple, separated by semicolon. Copy from specific mentions in the documentation, do not guess):
- "API_edit_tokens" - API keys with full edit, e.g. transaction execution access
- "Credit_card_number"
- "cyrpto_wallet_address"
- "passport_number"
- "..." - Any other sensitive data that allows deep execution access that is required as an input - specify what it is

STEP 10: Payment-specific analysis (only for servers with payment functionality):
- payments_analysis: Text field describing which data is used to process payments from the MCP inputSchema or tool parameters. Examples: "amount, recipient_address, payment_method", "invoice_id, card_token", "wallet_address, amount, gas_price". Leave empty if no payment processing capability.
- payments_autonomy: Rank the payment autonomy level (0-4):
  * 0: Not a payment processing server (no payment functionality)
  * 1: Only information about payments (invoice view, payment history, read-only)
  * 2: Payment request or link created (generates payment links/requests but doesn't execute)
  * 3: Payment processing via third-party (executes payments through external API like Stripe, PayPal)
  * 4: Payment processing directly (direct payment execution with full control, e.g., blockchain transactions)

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
    "action_space_description": "Read-only access to connected bank accounts via financial data aggregation APIs",
    "generality_industry": 0,
    "generality_environment": 0,
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
    "sensitive_data_required": "",
    "payments_analysis": "not related to payments",
    "payments_autonomy": 1
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
    "action_space_description": "Pre-defined blockchain operations on Base network via RPC endpoints",
    "generality_industry": 0,
    "generality_environment": 0,
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
    "sensitive_data_required": "private_key;rpc_endpoint",
    "payments_analysis": "wallet_address, private_key for signing already there, autonomous send_transaction tool without external approval",
    "payments_autonomy": 4
}

Example 3 - Poor documentation:
Input: ai-agent-mcp-servers
Description: Collection of MCP servers for AI agents
Tools: [No tools listed in documentation]
{
    "server": "ai-agent-mcp-servers",
    "analysis_notes": "Almost no detail in the Readme",
    "is_finance_llm": 0,
    "asset_type": "",
    "confidence": "L",
    "level": 1,
    "action_space_description": "Unclear - insufficient documentation to determine action space",
    "generality_industry": 1,
    "generality_environment": 1,
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
    "sensitive_data_required": "",
    "payments_analysis": "no data - assuming not for payments",
    "payments_autonomy": 0
}

Example 4 - General computer use:
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
    "action_space_description": "General-purpose file system access across any directory and file type, with arbitrary command execution",
    "generality_industry": 1,
    "generality_environment": 1,
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
    "sensitive_data_required": "",
    "payments_analysis": "not payment focused",
    "payments_autonomy": 0
}

Output Format:
{
    "server": "string",
    "analysis_notes": "Brief analysis of the tool(s)",
    "is_finance_llm": 0|1,
    "asset_type": "Banking|CreditCard|Crypto|Stocks|Accounting|Unclear",
    "confidence": "H|M|L",
    "level": 0|1|2|3|4|5,
    "action_space_description": "Description of what action space/environment this server operates in and its focus",
    "generality_industry": 0|1,
    "generality_environment": 0|1,
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
    "payments_analysis": "string describing payment data fields used",
    "payments_autonomy": 0|1|2|3|4
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
            "action_space_description",
            "generality_industry",
            "generality_environment",
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
            "sensitive_data_required",
            "payments_analysis",
            "payments_autonomy"
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
            "generality_industry",
            "generality_environment",
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

        # Validate payments_autonomy field (should be 0-4)
        if json_obj["payments_autonomy"] not in [0, 1, 2, 3, 4]:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Invalid payments_autonomy value: {json_obj['payments_autonomy']} (expected 0-4)"
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
    dataset_file = "../../data/internal-cl/clservers_input.jsonl"
    
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

# ============================================================================
# NAICS 3-DIGIT CLASSIFICATION TASK
# ============================================================================

NAICS_CLASSIFICATION_SYSTEM_PROMPT = """
ONLY RESPOND WITH JSON. NO MARKDOWN. NO TEXT OUTSIDE JSON.

You are classifying MCP (Model Context Protocol) servers by industry sector using the North American Industry Classification System (NAICS) 3-digit codes.

NAICS 3-DIGIT INDUSTRY CODES:

{naics_list}

CLASSIFICATION INSTRUCTIONS:

1. Analyze the MCP server name, description, tools, and documentation
2. Determine which NAICS 3-digit industry code best represents the primary use case or industry sector
3. If the server is truly cross-sector (usable across multiple industries with no clear primary focus), respond with "cross-sector"

OUTPUT FORMAT (JSON only):
{{
    "server": "server_name",
    "naics_code": "3-digit_code or cross-sector",
}}

EXAMPLES:

Example 1 - Banking/Finance Server:
Input: plaid-mcp
Description: Connect to bank accounts and retrieve financial data
Tools: get_accounts, get_transactions, get_balance
{{
    "server": "plaid-mcp",
    "naics_code": "522",
}}

Example 2 - Healthcare Server:
Input: fhir-mcp
Description: Access electronic health records using FHIR standard
Tools: get_patient, get_observation, search_conditions
{{
    "server": "fhir-mcp",
    "naics_code": "621",
}}

Example 3 - Cross-Sector Tool:
Input: filesystem-mcp
Description: Read and write files on local filesystem
Tools: read_file, write_file, list_directory, delete_file
{{
    "server": "filesystem-mcp",
    "naics_code": "cross-sector",
}}

RESPOND ONLY WITH JSON.
""".strip()


@scorer(metrics=[accuracy()])
def naics_classification_scorer() -> Scorer:
    """
    Custom scorer for validating NAICS classification JSON
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

        # Validate required fields
        required_fields = ["server", "naics_code", "reasoning"]
        missing_fields = [field for field in required_fields if field not in json_obj]

        if missing_fields:
            return Score(
                value=0,
                answer=completion,
                explanation=f"Missing required fields: {missing_fields}"
            )

        # Validate naics_code format (3 digits or "cross-sector")
        naics_code = str(json_obj["naics_code"])
        if naics_code != "cross-sector" and not (naics_code.isdigit() and len(naics_code) == 3):
            return Score(
                value=0,
                answer=completion,
                explanation=f"Invalid NAICS code: {naics_code} (must be 3-digit code or 'cross-sector')"
            )

        return Score(
            value=1,
            answer=completion,
            explanation="Valid NAICS classification"
        )

    return _scorer


@task
def naics_classification_task():
    """
    Inspect task for classifying MCP servers by NAICS 3-digit industry code
    Uses the same input dataset as finance_identification_task
    """
    dataset_file = "../../data/internal-cl/clservers_input.jsonl"

    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found")

    # Count samples in dataset to set appropriate message limit
    dataset_size = count_dataset_size(dataset_file)
    dynamic_message_limit = dataset_size + 10  # Add buffer for safety

    # Format NAICS list for prompt
    naics_list = format_naics_for_prompt()
    system_prompt = NAICS_CLASSIFICATION_SYSTEM_PROMPT.format(naics_list=naics_list)

    return Task(
        dataset=json_dataset(dataset_file),
        solver=[
            system_message(system_prompt),
            generate()
        ],
        scorer=[naics_classification_scorer()],
        message_limit=dynamic_message_limit
    )


# API key handling is managed automatically by AISI environment variables:
# - INSPECT_API_KEY_OVERRIDE=aisitools.api_key.override_api_key
# - ANTHROPIC_API_KEY (handled automatically by inspect framework)