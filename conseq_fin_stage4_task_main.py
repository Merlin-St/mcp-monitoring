#!/usr/bin/env python3
"""
mcp_onet_classify.py

End-to-end pipeline:
- Load MCP servers JSON (optionally finance-only)
- Extract tools -> build Inspect samples
- Define Inspect solver + task (L1 -> L2 -> Task ID)
- (Optional) run eval via `inspect eval this_file.py@task_name`
- Read logs with samples_df/messages_df and export one wide CSV

Usage:
  python conseq_fin_stage4_task_main.py --run (costs $500)
  or manually
  python conseq_fin_stage4_task_main.py \
    --onet onet_tasks.csv \
    --logs ./logs_mcp_onet \
    --out mcp_onet_classified.csv \
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
# analysis beta dataframes
from inspect_ai.analysis.beta import samples_df, messages_df  # noqa: F401  (messages_df imported per request)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger("mcp_onet")


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
    """Make Inspect samples: input is a JSON string with tool/server context."""
    samples: List[Dict[str, Any]] = []
    for t in tools:
        input_payload = {
            "tool_name": t["tool_name"],
            "tool_description": t["tool_description"],
            "tool_input_schema": t["tool_input_schema"],
            "server_name": t["server_name"],
            "server_description": t["server_description"],
            "readme_summary": t["readme_summary"],
        }
        samples.append({
            "input": json.dumps(input_payload, ensure_ascii=False),
            "target": "",  # generation-only; no ground truth
            "id": t["tool_id"],
            "metadata": {
                "stage": "onet_classification",
                "server_id": t["server_id"],
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


# ---------- Inspect solver ----------
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


@solver
def classify_tool(csv_path: str):
    rows = load_onet_rows(csv_path, None)
    L1, L2, L3 = group_onet_levels(rows)

    async def solve(state, generate):
        # parse the JSON we stuffed into Sample.input
        try:
            # Get input from the current sample
            input_text = getattr(state, 'input_text', None) or str(state.sample.input if hasattr(state, 'sample') else "")
            tool = json.loads(input_text)
        except Exception as e:
            # Fallback
            input_text = getattr(state, 'input_text', str(e))
            tool = {"tool_name": "", "tool_description": input_text}

        # Level 1
        p1 = (
            "Classify this MCP tool into an O*NET Level-1 cluster.\n\n"
            f"Tool: {tool.get('tool_name','')}\n"
            f"Desc: {tool.get('tool_description','')}\n"
            f"Server: {tool.get('server_name','')}\n"
            f"README: {tool.get('readme_summary','')}\n"
            f"InputSchema: {json.dumps(tool.get('tool_input_schema',''), ensure_ascii=False)}\n\n"
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

        # Also keep a single dict if you like
        state.metadata["final_solution"] = {
            "level1_cluster": l1_id, "level1_name": l1_name,
            "level2_cluster": l2_id, "level2_name": l2_name,
            "task_id": t_id, "task_name": t_name,
        }
        return state

    return solve


# ---------- Task definition (needs to be at module level for inspect to find it) ----------
@task
def mcp_onet_classify_task():
    # Use default paths that will be populated by main()
    samples_path = "conseq_fin_stage4_samples.jsonl"
    csv_path = "conseq_fin_stage4_tasks_cluster_names.csv"
    # No menu limits - all options always shown
    
    if not Path(samples_path).exists():
        raise FileNotFoundError(f"Samples file {samples_path} not found. Run with --run first.")
    
    return Task(
        dataset=json_dataset(samples_path),
        solver=chain(
            system_message("You are a careful classifier. Reply with ONLY the id each time."),
            classify_tool(csv_path),
        ),
    )


# ---------- Post-processing to CSV ----------
def export_csv(
    logs_dir: str,
    tools_json_path: str,
    out_csv_path: str,
) -> None:
    """
    Build a wide CSV with tool fields + chosen L1/L2/Task IDs and names.
    Only processes the most recent evaluation run.
    """
    # Find the most recent .eval file in the logs directory
    import glob
    from pathlib import Path
    
    eval_files = glob.glob(f"{logs_dir}/*.eval")
    if not eval_files:
        raise ValueError(f"No .eval files found in {logs_dir}")
    
    # Get the most recent eval file
    most_recent_eval = max(eval_files, key=lambda f: Path(f).stat().st_mtime)
    log.info(f"Processing most recent eval file: {most_recent_eval}")
    
    # Use samples_df with the specific directory
    df = samples_df(Path(most_recent_eval).parent)
    
    # Filter to only the most recent evaluation by eval file name
    eval_filename = Path(most_recent_eval).stem  # Remove .eval extension
    if 'eval' in df.columns:
        df = df[df['eval'].str.contains(eval_filename, na=False)]
    
    log.info(f"Found {len(df)} samples from most recent run: {eval_filename}")
    
    # id is the dataset id (tool_id)
    df = df.rename(columns={"id": "tool_id"})

    # load tool snapshot to enrich
    with open(tools_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    tools_map = {t["tool_id"]: t for t in tools}

    # build rows
    rows = []
    for _, r in df.iterrows():
        tid = r.get("tool_id")
        tinfo = tools_map.get(tid, {})
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
            # selections (expanded from metadata_* that we wrote)
            "level1_cluster": r.get("metadata_l1_id"),
            "level1_name": r.get("metadata_l1_name"),
            "level2_cluster": r.get("metadata_l2_id"),
            "level2_name": r.get("metadata_l2_name"),
            "task_id": r.get("metadata_task_id"),
            "task_name": r.get("metadata_task_name"),
            # optional: error/limit info for debugging
            "error": r.get("error"),
            "limit": r.get("limit"),
        })

    out_df = None
    try:
        import pandas as pd  # local import so script has minimal top deps
        out_df = pd.DataFrame(rows)
        Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv_path, index=False, encoding="utf-8")
    except Exception as e:
        raise
    finally:
        log.info("Wrote %d rows to %s", len(rows), out_csv_path)
    return


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", default="data_unified_filtered.json", help="Path to filtered MCP servers JSON")
    ap.add_argument("--onet", default="conseq_fin_stage4_tasks_cluster_names.csv", help="Path to O*NET tasks CSV")
    ap.add_argument("--logs", default=None, help="Directory to write/read Inspect logs (auto-generated if not provided)")
    ap.add_argument("--out", default="conseq_fin_stage4_task_output.csv", help="Output CSV path")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-20250514", help="Model string for inspect eval")
    ap.add_argument("--samples", default="conseq_fin_stage4_samples.jsonl", help="Samples JSONL path")
    ap.add_argument("--tools-json", default="conseq_fin_stage4_tools_full.json", help="Snapshot of tool records")
    ap.add_argument("--finance", action="store_true", help="Only finance-related servers (is_sector_52)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of samples processed by inspect eval")
    ap.add_argument("--run", action="store_true", help="Also launch `inspect eval` to produce logs")
    args = ap.parse_args()
    
    # Generate dated log directory if not provided
    if args.logs is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.logs = f"./logs_hierarchical_{timestamp}"
        log.info(f"Using auto-generated log directory: {args.logs}")

    # 1) Data prep
    servers = load_filtered_dataset(args.servers)
    tools = extract_tools_from_servers(servers, finance_only=args.finance)
    samples = create_inspect_samples_from_tools(tools)
    save_tools_json(tools, args.tools_json)
    save_samples_jsonl(samples, args.samples)

    # 2) Task is already defined at module level

    # 3) Optionally run inspect eval against this module’s task
    #    We call the CLI so scheduling/logging behaves exactly like Inspect intends.
    if args.run:
        Path(args.logs).mkdir(parents=True, exist_ok=True)
        # self-reference: this file path + @task_name
        this_file = Path(__file__).resolve()
        task_ref = f"{this_file}@mcp_onet_classify_task"
        cmd = [
            "inspect", "eval",
            task_ref,
            "--log-dir", args.logs,
            "--model", args.model,
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        log.info("Running: %s", " ".join(map(str, cmd)))
        subprocess.run(cmd, check=True)
    else:
        log.info("Not running eval automatically. To run manually, use:")
        this_file = Path(__file__).resolve()
        print(
            f"\ninspect eval {this_file}@mcp_onet_classify_task "
            f"--log-dir {args.logs} --model {args.model}\n"
        )

    # 4) Export results from logs → CSV (works whether you auto-ran or ran separately)
    export_csv(args.logs, args.tools_json, args.out)


if __name__ == "__main__":
    main()
