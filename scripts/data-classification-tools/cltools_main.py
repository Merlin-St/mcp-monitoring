#!/usr/bin/env python3
"""
mcp_onet_classify.py

End-to-end pipeline:
- Load MCP servers JSON (optionally finance-only)
- Extract tools -> build Inspect samples
- Define Inspect solvers + tasks:
    1) O*NET hierarchy (L1 -> L2 -> Task ID)
    2) Automation Level (0–5)
- (Optional) run eval via `inspect eval this_file.py@task_name` for each task
- Read logs with samples_df/messages_df and export one wide CSV

Usage:
  python cltools_main.py --run  # (costly)
  or manually:
  python cltools_main.py \
    --onet task_clusters_names.csv \
    --logs ./logs_onet \
    --logs-auto ./logs_auto \
    --out cltools_3_results.csv \
    --model openai/gpt-4o-mini \
    --finance \
    --limit 100 \
    --run
"""

import argparse
import csv
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- Inspect AI imports (runtime + analysis beta) ----
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import chain, solver, system_message
# analysis dataframes  
from inspect_ai.analysis.beta import samples_df, messages_df  # noqa: F401

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger("mcp_onet")

# ---------- Prompts ----------
AUTOMATION_LEVEL_PROMPT = """Your task is to classify this AI agent tool's (not the full mcp server - only the specific tool) automation level based on its execution capabilities:

0: Not a functional tool (e.g., just documentation or links)
1: MONITORING - Read-only access, information retrieval only
2: ANALYSIS - Processes and analyzes data, provides recommendations
3: META - Coordinates other tools/servers for execution
4: RESTRICTED EXECUTION - Modifies pre-specified external environments (e.g., executes transactions on specific systems)
5: UNRESTRICTED EXECUTION - Full execution capabilities (e.g., arbitrary code execution, unspecified modification capability across system/internet)

Examples:
1: get_account_balance
2: analyze_portfolio
3: run_daily_tasks
4: execute_trade
5: run_shell_command
ONLY REPLY WITH 0-5 or 'None' if you are unsure.
"""

FUNCTIONALITY_PROMPT = """Classify this MCP server tool by its primary function:

1. PERCEPTION (gathering information)
1.1 Sensors - database queries, monitoring, diagnostics, GUI reading, voice, search, physical sensing

2. REASONING (processing/analysis)
2.1 Planning - task decomposition, path-finding, workflow orchestration
2.2 Analysis - calculations, simulations, data processing
2.3 Resource Management - memory, self-management, resource allocation

3. ACTION (directly affecting the environment)
3.1 Authentication - login, CAPTCHA, wallet operations
3.2 Computer Use - GUI interaction, website automation, computer control
3.3 Code Execution - interpreters, IDE, file operations, running code
3.4 Software Extensions - calendar, social media APIs, third-party services
3.5 Physical Extensions - robotics, laboratory tools, physical world
3.6 Human Interaction - phone calls, messaging, direct communication
3.7 Agent Interaction - multi-agent coordination, sub-agents, third-party agents

Examples:
"get_database_records" → 1.1
"calculate_statistics" → 2.2
"execute_trade" → 3.4
"run_python_code" → 3.3

REPLY WITH NUMBER ONLY (e.g., 2.1) or 'None' if unclear.
"""

ACCESS_PATTERN_PROMPT = """Classify MCP tool by access pattern:

PERMISSIONS:
1: Read Only
2: Constrained Write (limited modifications)
3: Write (full modifications)

ENVIRONMENT:
A: Trusted (limited)
B: Untrusted (open)

MATRIX (Examples):
1A: Read/Trusted (RAG, database queries)
1B: Read/Untrusted (web scraping, deep research)
2A: Constrained/Trusted (specific API/GUI)
2B: Constrained/Untrusted (browser use)
3A: Write/Trusted (repository commits)
3B: Write/Untrusted (computer use with shell commands)

REPLY: CODE ONLY (e.g., 2A) or 'None'
"""


# ---------- Helper Functions (moved up for use in data prep) ----------
def _schema_str(schema) -> str:
    """Convert schema to string, returning empty for null-like values."""
    try:
        s = json.dumps(schema, ensure_ascii=False)
    except Exception:
        s = str(schema) if schema is not None else ""
    return "" if s in ("{}", "[]", "null", '""') else s


def _format_tool_context(tool: Dict[str, Any]) -> str:
    """
    Normalized context block for all prompts.
    Creates human-readable formatted text instead of JSON.
    """
    return (
        f"Tool Name: {tool.get('tool_name','')}\n"
        f"Tool Description & Input Schema: {tool.get('tool_description','')}\n {_schema_str(tool.get('tool_input_schema',''))}\n"
        f"Server name & Description & readme summary:\n"
        f"  - {tool.get('server_name','')}\n"
        f"  - {tool.get('server_description','')}\n"
        f"  - {tool.get('readme_summary','')}\n"
    )


# ---------- Data Prep ----------
def load_filtered_dataset(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded %d servers from %s", len(data), file_path)
    return data


def extract_tools_from_servers(servers: List[Dict[str, Any]], finance_only: bool = False) -> List[Dict[str, Any]]:
    """Flatten tool records with server context."""
    tools: List[Dict[str, Any]] = []
    for server in servers:
        if finance_only and not server.get("is_sector_52", False):
            continue
        server_tools = server.get("tools", []) or []
        if not server_tools:
            continue
        for i, tool in enumerate(server_tools):
            # Make tool_id unique by including index
            rec = {
                "tool_id": f"{server.get('id', 'unknown')}_{tool.get('name', 'unnamed')}_{i}",
                "tool_name": tool.get("name", ""),
                "tool_description": tool.get("description", ""),
                "tool_input_schema": tool.get("input_schema", {}),
                "server_id": server.get("id", ""),
                "server_name": server.get("name", ""),
                "server_description": server.get("canonical_description", ""),
                "readme_summary": server.get("readme_summary", ""),
                "server_data_sources": server.get("data_sources", []),
            }
            tools.append(rec)
    log.info("Extracted %d tools from %d servers", len(tools), len(servers))
    return tools


def create_inspect_samples_from_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Make Inspect samples: input is formatted text, metadata contains raw tool data."""
    samples: List[Dict[str, Any]] = []
    for t in tools:
        # Use formatted text instead of JSON
        ctx = _format_tool_context(t)
        samples.append({
            "input": ctx,  # Human-readable formatted text, NOT JSON
            "target": "",  # generation-only; no ground truth
            "id": t["tool_id"],
            "metadata": {
                "stage": "onet_classification",
                "server_id": t["server_id"],
                "tool_data": t  # Store complete tool dict for solvers
            }
        })
    return samples


def save_samples_jsonl(samples: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info("Wrote %d samples to %s", len(samples), path)


def save_tools_json(tools: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tools, f, indent=2, ensure_ascii=False)
    log.info("Wrote tools snapshot to %s", path)


# ---------- O*NET CSV helpers ----------
def load_onet_rows(csv_path: str, onet_code: Optional[str]) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if onet_code:
        rows = [r for r in rows if (r.get("O*NET-SOC Code") or "").strip() == onet_code]
    if not rows:
        raise ValueError("No O*NET rows matched the provided filter.")
    return rows


def group_onet_levels(rows: List[Dict[str, str]]):
    """
    Build 3 levels from your schema:
      L1: id=level1_cluster, name=level1_name
      L2: id=level2_cluster, name=level2_name, parent=level1_cluster
      L3: id=Task ID (numeric as string), name=Task, parent=level2_cluster
    """
    L1: Dict[str, Dict[str, str]] = {}
    L2: Dict[str, Tuple[str, str]] = {}  # id -> (name, parent_l1)
    L3: Dict[str, Tuple[str, str]] = {}  # id -> (name, parent_l2)

    for r in rows:
        l1 = (r.get("level1_cluster") or "").strip()
        l1n = (r.get("level1_name") or "").strip()
        l2 = (r.get("level2_cluster") or "").strip()
        l2n = (r.get("level2_name") or "").strip()
        t  = (r.get("Task ID") or "").strip()
        tn = (r.get("Task") or "").strip()

        if l1 and l1n:
            L1[l1] = {"id": l1, "name": l1n}
        if l2 and l2n and l1:
            L2[l2] = (l2n, l1)
        if t and tn and l2:
            L3[t]  = (tn, l2)

    if not L1 or not L2 or not L3:
        raise ValueError("O*NET CSV didn’t yield L1/L2/L3—check column names / filter.")
    return L1, L2, L3


def fmt_menu(d: Dict[str, Any]) -> str:
    """Format menu options - always shows ALL options (no limit)"""
    items = list(d.items())
    items.sort(key=lambda kv: kv[0])
    lines = []
    for k, v in items:
        if isinstance(v, dict):
            lines.append(f"{k}: {v.get('name','')}")
        elif isinstance(v, tuple):
            lines.append(f"{k}: {v[0]}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


# ---------- Inspect shared helpers ----------
async def _pick(state, generate, prompt: str, valid_ids: List[str], numeric_only: bool = False, retries: int = 2):
    for _ in range(retries + 1):
        state.messages.append(ChatMessageUser(content=prompt))
        state = await generate(state)  # base Inspect generate()
        choice = (state.output.completion or "").strip()
        if numeric_only and not choice.isdigit():
            continue
        if choice in valid_ids:
            return state, choice
    state.completed = True
    return state, None


def _parse_tool_from_state(state) -> Dict[str, Any]:
    """
    Parse tool data from state, preferring metadata.tool_data over JSON parsing.
    Supports both new format (tool_data in metadata) and legacy format (JSON in input).
    """
    # Priority 1: New format with metadata.tool_data
    if hasattr(state, "sample") and state.sample.metadata.get("tool_data"):
        return state.sample.metadata["tool_data"]
    
    # Priority 2: Legacy JSON format in input text
    try:
        input_text = getattr(state, 'input_text', None)
        if not input_text and hasattr(state, "sample"):
            input_text = getattr(state.sample, "input", "")
        if input_text:
            return json.loads(input_text)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    
    # Fallback: empty tool dict
    return {}


# ---------- Solver 1: O*NET hierarchical ----------
@solver
def classify_tool(csv_path: str):
    rows = load_onet_rows(csv_path, None)
    L1, L2, L3 = group_onet_levels(rows)

    async def solve(state, generate):
        # Level 1
        p1 = (
            "Classify this MCP tool into an O*NET Level-1 cluster.\n\n"
            f"Level-1 (id: name):\n{fmt_menu(L1)}\n\n"
            "Respond with ONLY the id."
        )
        state, l1_id = await _pick(state, generate, p1, list(L1.keys()))
        if state.completed:
            return state
        l1_name = L1[l1_id]["name"]

        # Level 2
        l2_opts = {k: v for k, v in L2.items() if v[1] == l1_id}
        if not l2_opts:
            state.completed = True
            return state
        p2 = (
            f"Level-1 chosen: {l1_id}: {l1_name}\n\n"
            f"Pick the SINGLE best Level-2 cluster id.\n"
            f"Level-2 (id: name):\n{fmt_menu(l2_opts)}\n\n"
            "Respond with ONLY the id."
        )
        state, l2_id = await _pick(state, generate, p2, list(l2_opts.keys()))
        if state.completed:
            return state
        l2_name = l2_opts[l2_id][0]

        # Level 3 (Task ID)
        l3_opts = {k: v for k, v in L3.items() if v[1] == l2_id}
        if not l3_opts:
            state.completed = True
            return state
        p3 = (
            f"Level-2 chosen: {l2_id}: {l2_name}\n\n"
            f"Pick the SINGLE best Level-3 Task ID (numeric).\n"
            f"Tasks (Task ID: Task):\n{fmt_menu(l3_opts)}\n\n"
            "Respond with ONLY the numeric Task ID."
        )
        state, t_id = await _pick(state, generate, p3, list(l3_opts.keys()), numeric_only=True)
        if state.completed:
            return state
        t_name = l3_opts[t_id][0]

        # Store flat metadata so samples_df expands columns nicely
        state.metadata["l1_id"] = l1_id
        state.metadata["l1_name"] = l1_name
        state.metadata["l2_id"] = l2_id
        state.metadata["l2_name"] = l2_name
        state.metadata["task_id"] = t_id
        state.metadata["task_name"] = t_name

        state.metadata["final_solution"] = {
            "level1_cluster": l1_id, "level1_name": l1_name,
            "level2_cluster": l2_id, "level2_name": l2_name,
            "task_id": t_id, "task_name": t_name,
        }
        return state

    return solve


# ---------- Solver 2: Automation Level ----------
@solver
def classify_automation_level():
    valid = set(list("012345"))

    async def solve(state, generate):
        prompt = AUTOMATION_LEVEL_PROMPT
        # ask up to 3 times for a single digit 0-5
        for _ in range(3):
            state.messages.append(ChatMessageUser(content=prompt))
            state = await generate(state)
            ans = (state.output.completion or "").strip()
            # keep only first digit to be safe
            ans_digit = "".join(ch for ch in ans if ch.isdigit())[:1]
            if ans_digit in valid:
                state.metadata["automation_level"] = ans_digit
                return state
            # tighten the instruction
            prompt = "ONLY REPLY WITH A SINGLE DIGIT NUMBER 0-5."
        # give up, mark as None-like
        state.metadata["automation_level"] = ""
        return state

    return solve


# ---------- Solver 3: Functionality ----------
@solver
def classify_functionality():
    # Valid functionality codes from the prompt
    valid = {"1.1", "2.1", "2.2", "2.3", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7"}

    async def solve(state, generate):
        prompt = FUNCTIONALITY_PROMPT
        # ask up to 3 times for a valid functionality code
        for _ in range(3):
            state.messages.append(ChatMessageUser(content=prompt))
            state = await generate(state)
            ans = (state.output.completion or "").strip()
            # Extract decimal number pattern (e.g., 1.1, 2.3, 3.7)
            import re
            match = re.search(r'\d+\.\d+', ans)
            if match:
                code = match.group()
                if code in valid:
                    state.metadata["functionality"] = code
                    return state
            # tighten the instruction
            prompt = "REPLY WITH NUMBER ONLY (e.g., 2.1) or 'None' if unclear."
        # give up, mark as None-like
        state.metadata["functionality"] = ""
        return state

    return solve


# ---------- Solver 4: Access Pattern ----------
@solver
def classify_access_pattern():
    # Valid access pattern codes from the prompt
    valid = {"1A", "1B", "2A", "2B", "3A", "3B"}

    async def solve(state, generate):
        prompt = ACCESS_PATTERN_PROMPT
        # ask up to 3 times for a valid access pattern code
        for _ in range(3):
            state.messages.append(ChatMessageUser(content=prompt))
            state = await generate(state)
            ans = (state.output.completion or "").strip().upper()
            # Extract pattern like 1A, 2B, 3A etc.
            import re
            match = re.search(r'[1-3][AB]', ans)
            if match:
                code = match.group()
                if code in valid:
                    state.metadata["access_pattern"] = code
                    return state
            # tighten the instruction
            prompt = "REPLY: CODE ONLY (e.g., 2A) or 'None'"
        # give up, mark as None-like
        state.metadata["access_pattern"] = ""
        return state

    return solve




# ---------- Tasks (module-level so inspect can find them) ----------
@task
def mcp_onet_classify_task():
    samples_path = "cltools_samples.jsonl"
    csv_path = "data/internal-task-clusters/task_clusters_names.csv"
    if not Path(samples_path).exists():
        raise FileNotFoundError(f"Samples file {samples_path} not found. Run with --run first.")
    return Task(
        dataset=json_dataset(samples_path),
        solver=chain(
            system_message("You are a careful classifier. Reply with ONLY the id each time."),
            classify_tool(csv_path),
        ),
    )


@task
def mcp_automation_level_task():
    samples_path = "cltools_samples.jsonl"
    if not Path(samples_path).exists():
        raise FileNotFoundError(f"Samples file {samples_path} not found. Run with --run first.")
    return Task(
        dataset=json_dataset(samples_path),
        solver=chain(
            system_message("Reply strictly as instructed. No extra words."),
            classify_automation_level(),
        ),
    )


@task
def mcp_functionality_task():
    samples_path = "cltools_samples.jsonl"
    if not Path(samples_path).exists():
        raise FileNotFoundError(f"Samples file {samples_path} not found. Run with --run first.")
    return Task(
        dataset=json_dataset(samples_path),
        solver=chain(
            system_message("You are a careful classifier. Reply with ONLY the code as instructed."),
            classify_functionality(),
        ),
    )


@task
def mcp_access_pattern_task():
    samples_path = "cltools_samples.jsonl"
    if not Path(samples_path).exists():
        raise FileNotFoundError(f"Samples file {samples_path} not found. Run with --run first.")
    return Task(
        dataset=json_dataset(samples_path),
        solver=chain(
            system_message("You are a careful classifier. Reply with ONLY the code as instructed."),
            classify_access_pattern(),
        ),
    )




# ---------- Post-processing to CSV ----------
def _read_latest_samples_df(logs_dir: str):
    """Read the most recent .eval file's samples into a DF."""
    import glob
    eval_files = glob.glob(f"{logs_dir}/*.eval")
    if not eval_files:
        return None, None
    most_recent_eval = max(eval_files, key=lambda f: Path(f).stat().st_mtime)
    df = samples_df(Path(most_recent_eval).parent)
    eval_filename = Path(most_recent_eval).stem
    if "eval" in df.columns:
        df = df[df["eval"].str.contains(eval_filename, na=False)]
    return df.rename(columns={"id": "tool_id"}), most_recent_eval


def export_csv(
    onet_logs_dir: str,
    auto_logs_dir: Optional[str],
    func_logs_dir: Optional[str],
    access_logs_dir: Optional[str],
    tools_json_path: str,
    out_csv_path: str,
) -> None:
    """
    Build a wide CSV with tool fields + chosen L1/L2/Task IDs/names
    + automation_level + functionality + access_pattern.
    """
    # Load base tool snapshot
    with open(tools_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    tools_map = {t["tool_id"]: t for t in tools}

    # Read ONET results
    onet_df, onet_eval_file = _read_latest_samples_df(onet_logs_dir)
    if onet_df is None:
        raise ValueError(f"No .eval files found in {onet_logs_dir}")
    log.info(f"Processing most recent O*NET eval: {onet_eval_file} ({len(onet_df)} samples)")

    # Read Automation results (optional)
    auto_df, auto_eval_file = (None, None)
    if auto_logs_dir and Path(auto_logs_dir).exists():
        auto_df, auto_eval_file = _read_latest_samples_df(auto_logs_dir)
        if auto_df is not None:
            log.info(f"Processing most recent Automation eval: {auto_eval_file} ({len(auto_df)} samples)")
        else:
            log.warning("No automation .eval files found; automation_level will be blank.")

    # Read Functionality results (optional)
    func_df, func_eval_file = (None, None)
    if func_logs_dir and Path(func_logs_dir).exists():
        func_df, func_eval_file = _read_latest_samples_df(func_logs_dir)
        if func_df is not None:
            log.info(f"Processing most recent Functionality eval: {func_eval_file} ({len(func_df)} samples)")
        else:
            log.warning("No functionality .eval files found; functionality will be blank.")

    # Read Access Pattern results (optional)
    access_df, access_eval_file = (None, None)
    if access_logs_dir and Path(access_logs_dir).exists():
        access_df, access_eval_file = _read_latest_samples_df(access_logs_dir)
        if access_df is not None:
            log.info(f"Processing most recent Access Pattern eval: {access_eval_file} ({len(access_df)} samples)")
        else:
            log.warning("No access pattern .eval files found; access_pattern will be blank.")


    # Build quick lookup maps from metadata
    def meta_lookup(df: Optional["pd.DataFrame"], keys: List[str]) -> Dict[str, Dict[str, Any]]:
        if df is None:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for _, r in df.iterrows():
            tid = r.get("tool_id")
            if not tid:
                continue
            out[tid] = {k: r.get(f"metadata_{k}") for k in keys}
        return out

    # pandas import here to satisfy type
    import pandas as pd  # noqa: F401
    onet_meta = meta_lookup(onet_df, ["l1_id", "l1_name", "l2_id", "l2_name", "task_id", "task_name"])
    auto_meta = meta_lookup(auto_df, ["automation_level"])
    func_meta = meta_lookup(func_df, ["functionality"])
    access_meta = meta_lookup(access_df, ["access_pattern"])

    # Helper functions to parse classification codes
    def parse_automation_level(code: str) -> str:
        """Parse automation level code (0-5) into descriptive text."""
        if not code or code == "":
            return ""
        
        automation_map = {
            "0": "0 (Not a tool)",
            "1": "1 (Monitoring)",
            "2": "2 (Analysis)",
            "3": "3 (Meta-coordination)",
            "4": "4 (Restricted execution)",
            "5": "5 (Unrestricted execution)"
        }
        
        return automation_map.get(code, code)
    
    def parse_functionality(code: str) -> Tuple[str, str]:
        """Parse functionality code like '2.1' into main and sub categories."""
        if not code or code == "":
            return "", ""
        
        # Mapping for main categories
        main_map = {
            "1": "perception",
            "2": "reasoning", 
            "3": "action"
        }
        
        # Mapping for sub categories
        sub_map = {
            "1.1": "sensors",
            "2.1": "planning",
            "2.2": "analysis",
            "2.3": "resource_management",
            "3.1": "authentication",
            "3.2": "computer_use",
            "3.3": "code_execution",
            "3.4": "software_extensions",
            "3.5": "physical_extensions",
            "3.6": "human_interaction",
            "3.7": "agent_interaction"
        }
        
        # Extract main category from first digit
        main = main_map.get(code.split(".")[0], "") if "." in code else ""
        # Get full sub category
        sub = sub_map.get(code, "")
        
        return main, sub
    
    def parse_access_pattern(code: str) -> Tuple[str, str]:
        """Parse access pattern code like '2A' into permissions and environment."""
        if not code or code == "":
            return "", ""
        
        # Mapping for permissions (first character)
        perm_map = {
            "1": "read",
            "2": "constrained_write",
            "3": "write"
        }
        
        # Mapping for environment (second character)
        env_map = {
            "A": "trusted",
            "B": "untrusted"
        }
        
        # Parse the code
        permissions = perm_map.get(code[0], "") if len(code) > 0 else ""
        environment = env_map.get(code[1], "") if len(code) > 1 else ""
        
        return permissions, environment

    # Build rows
    rows = []
    for tid, tinfo in tools_map.items():
        onet = onet_meta.get(tid, {})
        auto = auto_meta.get(tid, {})
        func = func_meta.get(tid, {})
        access = access_meta.get(tid, {})
        
        # Parse all classification codes
        auto_code = auto.get("automation_level", "")
        auto_parsed = parse_automation_level(auto_code)
        
        func_code = func.get("functionality", "")
        func_main, func_sub = parse_functionality(func_code)
        
        access_code = access.get("access_pattern", "")
        access_perms, access_env = parse_access_pattern(access_code)
        
        rows.append({
            # original fields
            "tool_id": tid,
            "tool_name": tinfo.get("tool_name", ""),
            "tool_description": tinfo.get("tool_description", ""),
            "tool_input_schema": json.dumps(tinfo.get("tool_input_schema", {}), ensure_ascii=False),
            "server_id": tinfo.get("server_id", ""),
            "server_name": tinfo.get("server_name", ""),
            "server_description": tinfo.get("server_description", ""),
            "readme_summary": tinfo.get("readme_summary", ""),
            "server_data_sources": json.dumps(tinfo.get("server_data_sources", []), ensure_ascii=False),
            # O*NET selections
            "level1_cluster": onet.get("l1_id"),
            "level1_name": onet.get("l1_name"),
            "level2_cluster": onet.get("l2_id"),
            "level2_name": onet.get("l2_name"),
            "task_id": onet.get("task_id"),
            "task_name": onet.get("task_name"),
            # Classification fields
            "automation_level": auto_parsed,
            # Functionality - parsed into main and sub
            "tool_functionality_main": func_main,
            "tool_functionality_sub": func_sub,
            # Access pattern - parsed into permissions and environment
            "tool_access_permissions": access_perms,
            "tool_access_environment": access_env,
        })

    # Write CSV
    import pandas as pd
    out_df = pd.DataFrame(rows)
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False, encoding="utf-8")
    log.info("Wrote %d rows to %s", len(rows), out_csv_path)


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", default="data/initial/data_unified_filtered.json", help="Path to filtered MCP servers JSON")
    ap.add_argument("--onet", default="data/internal-task-clusters/task_clusters_names.csv", help="Path to O*NET tasks CSV")
    ap.add_argument("--logs", default=None, help="Log dir for O*NET task (auto-generated if not provided)")
    ap.add_argument("--logs-auto", default=None, help="Log dir for Automation Level task")
    ap.add_argument("--logs-func", default=None, help="Log dir for Functionality task")
    ap.add_argument("--logs-access", default=None, help="Log dir for Access Pattern task")
    ap.add_argument("--out", default="data/internal-cl/cltools_3_results.csv", help="Output CSV path")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-20250514", help="Model string for inspect eval")
    ap.add_argument("--samples", default="cltools_samples.jsonl", help="Samples JSONL path")
    ap.add_argument("--tools-json", default="cltools_prep.json", help="Snapshot of tool records")
    ap.add_argument("--finance", action="store_true", help="Only finance-related servers (is_sector_52)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of samples processed by inspect eval")
    ap.add_argument("--max-connections", type=int, default=None, help="Maximum number of concurrent connections for inspect eval")
    ap.add_argument("--run", action="store_true", help="Also launch `inspect eval` to produce logs for all tasks")
    args = ap.parse_args()

    # Generate dated log directories if not provided - all under .log parent directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.logs is None:
        args.logs = f"./logs/onet_{timestamp}"
        log.info(f"Using auto-generated O*NET log directory: {args.logs}")
    if args.logs_auto is None:
        args.logs_auto = f"./.log/auto_{timestamp}"
        log.info(f"Using auto-generated Automation log directory: {args.logs_auto}")
    if args.logs_func is None:
        args.logs_func = f"./.log/func_{timestamp}"
        log.info(f"Using auto-generated Functionality log directory: {args.logs_func}")
    if args.logs_access is None:
        args.logs_access = f"./.log/access_{timestamp}"
        log.info(f"Using auto-generated Access Pattern log directory: {args.logs_access}")

    # 1) Data prep
    servers = load_filtered_dataset(args.servers)
    tools = extract_tools_from_servers(servers, finance_only=args.finance)
    samples = create_inspect_samples_from_tools(tools)
    save_tools_json(tools, args.tools_json)
    save_samples_jsonl(samples, args.samples)

    # 2) Tasks are already defined at module level

    # 3) Optionally run inspect eval for ALL four tasks
    if args.run:
        for which, logs_dir, task_name in [
            ("O*NET", args.logs, "mcp_onet_classify_task"),
            ("Automation", args.logs_auto, "mcp_automation_level_task"),
            ("Functionality", args.logs_func, "mcp_functionality_task"),
            ("Access Pattern", args.logs_access, "mcp_access_pattern_task"),
        ]:
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            this_file = Path(__file__).resolve()
            task_ref = f"{this_file}@{task_name}"
            cmd = [
                "inspect", "eval",
                task_ref,
                "--log-dir", logs_dir,
                "--model", args.model,
            ]
            if args.limit:
                cmd.extend(["--limit", str(args.limit)])
            if args.max_connections:
                cmd.extend(["--max-connections", str(args.max_connections)])
            log.info("Running %s task: %s", which, " ".join(map(str, cmd)))
            subprocess.run(cmd, check=True)
    else:
        this_file = Path(__file__).resolve()
        log.info("Not running eval automatically. To run manually, use:")
        print(
            f"\ninspect eval {this_file}@mcp_onet_classify_task "
            f"--log-dir {args.logs} --model {args.model}\n"
            f"inspect eval {this_file}@mcp_automation_level_task "
            f"--log-dir {args.logs_auto} --model {args.model}\n"
            f"inspect eval {this_file}@mcp_functionality_task "
            f"--log-dir {args.logs_func} --model {args.model}\n"
            f"inspect eval {this_file}@mcp_access_pattern_task "
            f"--log-dir {args.logs_access} --model {args.model}\n"
        )

    # 4) Export merged results (only if eval files exist)
    try:
        export_csv(args.logs, args.logs_auto, args.logs_func, args.logs_access, args.tools_json, args.out)
    except ValueError as e:
        if "No .eval files found" in str(e):
            log.info("No evaluation results found yet. Run evaluations first to generate CSV output.")
        else:
            raise


if __name__ == "__main__":
    main()
