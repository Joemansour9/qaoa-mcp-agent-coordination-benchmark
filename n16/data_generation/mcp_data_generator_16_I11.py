"""
MCP Agent Interaction Data Generator — 16 Node Version — INSTANCE I11
======================================================================
Generates the 16x16 cost matrix for Instance I11: a sequential-chain
MCP tool-interaction graph, extending the I6-I10 topology-shift study.

Identical pipeline to mcp_data_generator_16.py (same MCP tool
definitions, same GPT-4o-mini agent harness, same execute_tool
simulation, same cost-matrix construction) — the ONLY change is the
TASKS dictionary below.

Task-mix design for I11 — SEQUENTIAL CHAIN (8-server cycle):
  - Fix a cyclic server order: FileSystem -> WebSearch -> Database ->
    Analytics -> Email -> Calendar -> CodeExecution -> VectorSearch ->
    (back to FileSystem).
  - Every task is a deterministic 3-step chain touching 3 consecutive
    servers in this cycle (e.g. FileSystem -> WebSearch -> Database,
    or WebSearch -> Database -> Analytics, etc). Task slots are
    assigned to one of the 8 possible chain-start positions via
    round-robin (i % 8) — same mechanism as I9's spoke assignment,
    repurposed for a PATH pattern instead of a STAR pattern.
  - No task jumps more than 2 steps ahead in the cycle, so a server
    should mainly connect to its immediate neighbors rather than to
    the whole graph (star) or to one fixed partner (pairwise
    concentration) or being disconnected (isolation).

Expected effect on the resulting cost matrix: E_I11 should approximate
a ring/path graph — each server's edges concentrated on its 1-2
neighbors in the cycle, with roughly uniform, moderate edge weights
around the ring rather than one dominant edge or one hub — a fourth
distinct edge-SET divergence pattern (I6 = isolation, I7/I8 = pairwise
concentration, I9 = star, I11 = ring/path).

Paper: "Benchmarking QAOA on MCP-Derived Agent Coordination
        Instances: Scaling from n=8 to n=16 on IBM Hardware"

Design:
  - 1 orchestrating LLM agent (GPT-4o-mini)
  - 8 MCP servers, 2 tools each = 16 nodes (matches 16-qubit QAOA circuit)
  - 150 tasks across 3 complexity levels (50 each)
  - Captures: tokens per tool call, tool-to-tool transitions
  - Output: 16x16 cost matrix CSV + raw interaction log

MCP Servers (8 servers × 2 tools = 16 nodes):
  Server A — FileSystem:      [read_file, write_file]         nodes 0,1
  Server B — WebSearch:       [search_web, fetch_url]         nodes 2,3
  Server C — Database:        [query_db, insert_record]       nodes 4,5
  Server D — Analytics:       [run_analysis, generate_report] nodes 6,7
  Server E — Email:           [send_email, read_inbox]        nodes 8,9
  Server F — Calendar:        [create_event, list_events]     nodes 10,11
  Server G — CodeExecution:   [run_code, get_output]          nodes 12,13
  Server H — VectorSearch:    [embed_query, search_index]     nodes 14,15

Estimated cost: ~$0.15 for 150 tasks

Setup:
  pip install openai python-dotenv pandas numpy
  Create .env file: OPENAI_API_KEY=your_key_here

Usage:
  python mcp_data_generator_16_I11.py --tasks 150 --output mcp_agent_data_16_I11.csv
"""

import os
import json
import time
import random
import argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    raise ImportError("Run: pip install openai python-dotenv pandas numpy")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  16-NODE MCP TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

MCP_TOOLS = [
    {"type": "function", "function": {"name": "filesystem_read_file",
        "description": "MCP FileSystem Server: Read a file and return its contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to read"}},
            "required": ["path"]}}},
    {"type": "function", "function": {"name": "filesystem_write_file",
        "description": "MCP FileSystem Server: Write content to a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "websearch_search_web",
        "description": "MCP WebSearch Server: Search the web for information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"}},
            "required": ["query"]}}},
    {"type": "function", "function": {"name": "websearch_fetch_url",
        "description": "MCP WebSearch Server: Fetch content from a specific URL.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "URL to fetch"}},
            "required": ["url"]}}},
    {"type": "function", "function": {"name": "database_query_db",
        "description": "MCP Database Server: Execute a database query and return results.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "SQL query"},
            "database": {"type": "string", "description": "Database name"}},
            "required": ["query", "database"]}}},
    {"type": "function", "function": {"name": "database_insert_record",
        "description": "MCP Database Server: Insert a record into a database table.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string", "description": "Table name"},
            "record": {"type": "object", "description": "Record to insert"}},
            "required": ["table", "record"]}}},
    {"type": "function", "function": {"name": "analytics_run_analysis",
        "description": "MCP Analytics Server: Run statistical analysis on provided data.",
        "parameters": {"type": "object", "properties": {
            "data": {"type": "string", "description": "Data to analyze (JSON string)"},
            "analysis_type": {"type": "string", "description": "Type: summary|correlation|trend"}},
            "required": ["data", "analysis_type"]}}},
    {"type": "function", "function": {"name": "analytics_generate_report",
        "description": "MCP Analytics Server: Generate a formatted report from analysis results.",
        "parameters": {"type": "object", "properties": {
            "analysis_results": {"type": "string", "description": "Results to report"},
            "format": {"type": "string", "description": "Output format: markdown|json|text"}},
            "required": ["analysis_results", "format"]}}},
    {"type": "function", "function": {"name": "email_send_email",
        "description": "MCP Email Server: Send an email to specified recipients.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body content"}},
            "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "email_read_inbox",
        "description": "MCP Email Server: Read emails from inbox with optional filtering.",
        "parameters": {"type": "object", "properties": {
            "folder": {"type": "string", "description": "Email folder to read"},
            "limit": {"type": "integer", "description": "Maximum emails to return"}},
            "required": ["folder"]}}},
    {"type": "function", "function": {"name": "calendar_create_event",
        "description": "MCP Calendar Server: Create a new calendar event.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "Event title"},
            "start_time": {"type": "string", "description": "Event start time (ISO format)"},
            "duration_minutes": {"type": "integer", "description": "Duration in minutes"},
            "attendees": {"type": "string", "description": "Comma-separated attendee emails"}},
            "required": ["title", "start_time"]}}},
    {"type": "function", "function": {"name": "calendar_list_events",
        "description": "MCP Calendar Server: List calendar events within a date range.",
        "parameters": {"type": "object", "properties": {
            "start_date": {"type": "string", "description": "Start date (ISO format)"},
            "end_date": {"type": "string", "description": "End date (ISO format)"}},
            "required": ["start_date", "end_date"]}}},
    {"type": "function", "function": {"name": "code_run_code",
        "description": "MCP CodeExecution Server: Execute a code snippet and return results.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Code to execute"},
            "language": {"type": "string", "description": "Programming language: python|javascript|bash"}},
            "required": ["code", "language"]}}},
    {"type": "function", "function": {"name": "code_get_output",
        "description": "MCP CodeExecution Server: Retrieve output from a previous code execution.",
        "parameters": {"type": "object", "properties": {
            "execution_id": {"type": "string", "description": "Execution ID to retrieve output for"}},
            "required": ["execution_id"]}}},
    {"type": "function", "function": {"name": "vector_embed_query",
        "description": "MCP VectorSearch Server: Generate embeddings for a text query.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Text to embed"},
            "model": {"type": "string", "description": "Embedding model to use"}},
            "required": ["text"]}}},
    {"type": "function", "function": {"name": "vector_search_index",
        "description": "MCP VectorSearch Server: Search a vector index for similar documents.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"},
            "index_name": {"type": "string", "description": "Vector index to search"},
            "top_k": {"type": "integer", "description": "Number of results to return"}},
            "required": ["query", "index_name"]}}},
]

TOOL_TO_NODE = {
    "filesystem_read_file":      0,
    "filesystem_write_file":     1,
    "websearch_search_web":      2,
    "websearch_fetch_url":       3,
    "database_query_db":         4,
    "database_insert_record":    5,
    "analytics_run_analysis":    6,
    "analytics_generate_report": 7,
    "email_send_email":          8,
    "email_read_inbox":          9,
    "calendar_create_event":     10,
    "calendar_list_events":      11,
    "code_run_code":             12,
    "code_get_output":           13,
    "vector_embed_query":        14,
    "vector_search_index":       15,
}

TOOL_TO_SERVER = {
    "filesystem_read_file":      "FileSystem",
    "filesystem_write_file":     "FileSystem",
    "websearch_search_web":      "WebSearch",
    "websearch_fetch_url":       "WebSearch",
    "database_query_db":         "Database",
    "database_insert_record":    "Database",
    "analytics_run_analysis":    "Analytics",
    "analytics_generate_report": "Analytics",
    "email_send_email":          "Email",
    "email_read_inbox":          "Email",
    "calendar_create_event":     "Calendar",
    "calendar_list_events":      "Calendar",
    "code_run_code":             "CodeExecution",
    "code_get_output":           "CodeExecution",
    "vector_embed_query":        "VectorSearch",
    "vector_search_index":       "VectorSearch",
}

N_NODES = 16

# Fixed 8-server cycle: task chains walk 3 consecutive servers in this order.
CYCLE = ["filesystem", "websearch", "database", "analytics",
         "email", "calendar", "codeexec", "vectorsearch"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TASK DEFINITIONS — SEQUENTIAL CHAIN (one pool per chain-start server)
# ═══════════════════════════════════════════════════════════════════════════════

TASKS = {
    "simple": {
        "filesystem":   ["Read the file server_list.txt, search the web for each server's latest status page, and query the database to log which ones are outdated"],
        "websearch":    ["Search the web for current market prices, query the database for our current holdings, and run an analysis comparing the two"],
        "database":     ["Query the sales database for this month's totals, run an analysis of the trend, and email the summary to the team"],
        "analytics":    ["Run an analysis of last week's usage data, email the findings to the team, and create a calendar event to review it"],
        "email":        ["Read the inbox for the bug report, create a calendar event for the triage meeting, and run code to reproduce the reported issue"],
        "calendar":     ["List this week's calendar events, run code to extract the meeting topics, and embed the topics for future semantic search"],
        "codeexec":     ["Run a Python script to generate a summary, search the vector index for related past summaries, and write the combined result to a file"],
        "vectorsearch": ["Search the vector index for prior notes on this topic, read the matching file, and search the web for anything more recent"],
    },
    "medium": {
        "filesystem":   ["Read the deployment manifest file, search the web for the changelog of each listed dependency, and query the database to record which dependencies are behind"],
        "websearch":    ["Search the web for two competitor benchmark reports, query the database for our matching internal metrics, and run a comparative analysis"],
        "database":     ["Query the support database for tickets from the last quarter, run an analysis of resolution times, and email a summary report to the support lead"],
        "analytics":    ["Run a variance analysis on the quarterly figures, email the results to leadership, and create calendar events for the follow-up discussion"],
        "email":        ["Read the inbox for incident reports from the last 24 hours, create calendar events for each incident's postmortem, and run code to generate a summary script for each"],
        "calendar":     ["List next month's scheduled reviews, run code to parse the review agendas, and search the vector index for related prior review notes"],
        "codeexec":     ["Run code to compute this sprint's velocity metrics, search the vector index for similar past sprints, and write a comparison file to disk"],
        "vectorsearch": ["Search the vector index for related past incident reports, read the top matching file in full, and search the web for any newly published advisories on the same issue"],
    },
    "complex": {
        "filesystem":   ["Read every config file in the release bundle, search the web for security advisories matching each listed library, and query the database to cross-reference which of our deployed instances are affected"],
        "websearch":    ["Search the web for the latest industry compliance thresholds, query the database for every record that might violate them, and run a full statistical analysis of the exposure"],
        "database":     ["Query the compliance database for all flagged accounts, run a full risk analysis across every flagged record, and email a detailed findings report to the compliance team"],
        "analytics":    ["Run a full statistical analysis reconciling three separate data sources, email a detailed report to all stakeholders, and create calendar events for each stakeholder's individual review session"],
        "email":        ["Read the inbox for every escalation this week, create calendar events for each escalation's resolution meeting, and run code to build a consolidated triage dashboard from the escalation data"],
        "calendar":     ["List every calendar event across the next quarter, run code to categorize each event by project, and embed the full categorized schedule into the vector index for future retrieval"],
        "codeexec":     ["Run a full code-based extraction of this release's changelog, search the vector index for related historical release notes, and write a consolidated release-history file to disk"],
        "vectorsearch": ["Search the vector index for every related past audit finding, read each matching file, and search the web for the latest regulatory guidance on each finding"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SIMULATED TOOL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def execute_tool(tool_name: str, arguments: dict) -> str:
    """Return realistic simulated tool response for all 16 tools."""
    responses = {
        "filesystem_read_file":      f"File contents of '{arguments.get('path', 'unknown')}': [Structured file content with relevant data for current task.]",
        "filesystem_write_file":     f"Successfully wrote {len(str(arguments.get('content', '')))} characters to '{arguments.get('path', 'unknown')}'.",
        "websearch_search_web":      f"Search results for '{arguments.get('query', '')}': [5 relevant results with key findings, documentation, and recent analysis.]",
        "websearch_fetch_url":       f"Content from {arguments.get('url', 'URL')}: [Webpage content with structured data and relevant information.]",
        "database_query_db":         f"Query results from {arguments.get('database', 'db')}: [3 records returned with id, value, status fields.]",
        "database_insert_record":    f"Successfully inserted record into '{arguments.get('table', 'table')}'. New record ID: {np.random.randint(1000, 9999)}",
        "analytics_run_analysis":    f"Analysis complete ({arguments.get('analysis_type', 'summary')}): Mean=42.3, Std=8.7, Trend=+12.4%, Correlation=0.87.",
        "analytics_generate_report": f"Report generated in {arguments.get('format', 'markdown')} format: Executive Summary | Key Findings | Recommendations. ID: RPT-{np.random.randint(100, 999)}",
        "email_send_email":          f"Email sent successfully to {arguments.get('to', 'recipient')}. Subject: '{arguments.get('subject', '')}'. Message ID: MSG-{np.random.randint(1000, 9999)}",
        "email_read_inbox":          f"Inbox '{arguments.get('folder', 'INBOX')}': [3 emails found. Latest: 'Re: Project Update' from team@company.com, received 2 hours ago.]",
        "calendar_create_event":     f"Event created: '{arguments.get('title', 'Meeting')}' scheduled for {arguments.get('start_time', 'TBD')}. Event ID: EVT-{np.random.randint(100, 999)}",
        "calendar_list_events":      f"Events from {arguments.get('start_date', 'today')} to {arguments.get('end_date', 'next week')}: [3 events found: Team Standup, Sprint Review, Client Call.]",
        "code_run_code":             f"Code executed successfully ({arguments.get('language', 'python')}). Execution ID: EXEC-{np.random.randint(1000, 9999)}. Output: [Results computed and ready.]",
        "code_get_output":           f"Output for execution {arguments.get('execution_id', 'EXEC-0000')}: [Execution completed. Return value: 42. Stdout: 'Process complete. 150 records processed.']",
        "vector_embed_query":        f"Embedding generated for '{arguments.get('text', '')[:50]}...'. Vector dimension: 1536. Embedding ID: EMB-{np.random.randint(1000, 9999)}",
        "vector_search_index":       f"Search results from index '{arguments.get('index_name', 'default')}': [Top 3 matches found with similarity scores 0.94, 0.87, 0.82.]",
    }
    return responses.get(tool_name, f"Tool {tool_name} executed successfully.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  AGENT EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent_task(task: str, complexity: str, task_id: int) -> dict:
    """Run a single task and collect interaction metrics."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI agent with access to MCP (Model Context Protocol) tools "
                "across eight servers: FileSystem, WebSearch, Database, Analytics, "
                "Email, Calendar, CodeExecution, and VectorSearch. "
                "Complete the given task by calling the appropriate tools in a logical sequence. "
                "Use multiple tools as needed. Always complete the full task."
            )
        },
        {"role": "user", "content": task}
    ]

    tool_sequence    = []
    tool_transitions = []
    tokens_per_tool  = {}
    latency_per_tool = {}
    total_tokens     = 0
    prev_tool        = None
    max_iterations   = 15
    iteration        = 0
    success          = False

    try:
        while iteration < max_iterations:
            iteration += 1
            t_start = time.time()

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=MCP_TOOLS,
                tool_choice="auto",
                max_tokens=500
            )

            latency_ms   = (time.time() - t_start) * 1000
            step_tokens  = response.usage.total_tokens
            total_tokens += step_tokens
            msg          = response.choices[0].message

            if not msg.tool_calls:
                success = True
                break

            tool_results = []
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                tool_sequence.append(tool_name)
                per_tool_tokens = step_tokens // max(len(msg.tool_calls), 1)
                tokens_per_tool[tool_name]  = tokens_per_tool.get(tool_name, 0)  + per_tool_tokens
                latency_per_tool[tool_name] = latency_per_tool.get(tool_name, 0) + latency_ms

                if prev_tool is not None:
                    tool_transitions.append((prev_tool, tool_name))
                prev_tool = tool_name

                result = execute_tool(tool_name, arguments)
                tool_results.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": result
                })

            messages.append(msg)
            messages.extend(tool_results)

        if iteration >= max_iterations:
            success = len(tool_sequence) > 0

    except Exception as e:
        print(f"  Error on task {task_id}: {e}")
        success = False

    return {
        "task_id":          task_id,
        "task":             task[:100],
        "complexity":       complexity,
        "tool_sequence":    tool_sequence,
        "tool_transitions": tool_transitions,
        "tokens_per_tool":  tokens_per_tool,
        "latency_per_tool": latency_per_tool,
        "total_tokens":     total_tokens,
        "n_tool_calls":     len(tool_sequence),
        "success":          success
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  COST MATRIX CONSTRUCTION  (16×16)
# ═══════════════════════════════════════════════════════════════════════════════

def build_cost_matrix(results: list) -> np.ndarray:
    """Build 16×16 cost matrix from agent interaction records."""
    n = N_NODES
    transition_costs = [[[] for _ in range(n)] for _ in range(n)]

    for r in results:
        if not r['success']:
            continue
        transitions   = r['tool_transitions']
        total_tok     = r['total_tokens']
        n_transitions = max(len(transitions), 1)
        cost_per_step = total_tok / n_transitions

        for (t1, t2) in transitions:
            i = TOOL_TO_NODE.get(t1)
            j = TOOL_TO_NODE.get(t2)
            if i is not None and j is not None:
                transition_costs[i][j].append(cost_per_step)

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if transition_costs[i][j]:
                W[i][j] = np.mean(transition_costs[i][j])
    return W


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def allocate_chain_slots(per_complexity: int, cycle: list):
    """
    Deterministically assigns each of the per_complexity task slots in a
    tier to one of the 8 chain-start servers via round-robin
    (i % len(cycle)) — no RNG involved in WHICH chain-start a slot gets,
    fully reproducible by inspection. This spreads chain-starts evenly
    around the cycle rather than concentrating on any one server.
    """
    return [cycle[i % len(cycle)] for i in range(per_complexity)]


def run_experiment(n_tasks: int, output_path: str):
    per_complexity = n_tasks // 3
    chain_assignment = allocate_chain_slots(per_complexity, CYCLE)
    from collections import Counter
    chain_counts = Counter(chain_assignment)

    print(f"\n{'='*60}")
    print(f"MCP 16-Node Agent Data Generation — INSTANCE I11")
    print(f"Servers: 8 × 2 tools = 16 nodes")
    print(f"Tasks:   {n_tasks} ({per_complexity} per complexity level)")
    print(f"Model:   gpt-4o-mini")
    print(f"Est cost: ~${n_tasks * 0.001:.2f}")
    print(f"Sequential-chain design: every task walks 3 consecutive "
          f"servers in the cycle {CYCLE}, round-robin start position")
    print(f"  Per-tier chain-start counts (deterministic): {dict(chain_counts)}")
    print(f"{'='*60}\n")

    all_results = []
    task_id     = 0
    random.seed(602120)  # only affects WHICH prompt variant is picked
                          # within a chain-start's pool, never which
                          # chain-start a task slot gets (round-robin above)
    chain_count_by_tier = {}

    for complexity in ['simple', 'medium', 'complex']:
        pools = TASKS[complexity]
        print(f"\n── {complexity.upper()} tasks ({per_complexity}) ──")

        tier_chain_counts = Counter()
        for i in range(per_complexity):
            chain_start = chain_assignment[i]
            task = random.choice(pools[chain_start])
            tier_chain_counts[chain_start] += 1
            task_id += 1
            print(f"  [{task_id:3d}/{n_tasks}] [{chain_start[:4].upper():4s}] {task[:50]}...",
                  end=' ', flush=True)
            result = run_agent_task(task, complexity, task_id)
            all_results.append(result)
            status = "✓" if result['success'] else "✗"
            print(f"{status} | tools={result['n_tool_calls']} "
                  f"| tokens={result['total_tokens']}")
            time.sleep(0.3)

        chain_count_by_tier[complexity] = dict(tier_chain_counts)
        print(f"  (chain-start distribution this tier: {dict(tier_chain_counts)})")

    print(f"\nOverall chain-start distribution: {dict(chain_counts)} "
          f"(target: even split across {len(CYCLE)} positions)")

    # Save raw data
    rows = []
    for r in all_results:
        rows.append({
            'task_id':          r['task_id'],
            'task':             r['task'],
            'complexity':       r['complexity'],
            'tool_sequence':    json.dumps(r['tool_sequence']),
            'tool_transitions': json.dumps(r['tool_transitions']),
            'tokens_per_tool':  json.dumps(r['tokens_per_tool']),
            'latency_per_tool': json.dumps(r['latency_per_tool']),
            'total_tokens':     r['total_tokens'],
            'n_tool_calls':     r['n_tool_calls'],
            'success':          r['success']
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nRaw data saved → {output_path}")

    # Build and save cost matrix
    W          = build_cost_matrix(all_results)
    tool_names = list(TOOL_TO_NODE.keys())
    matrix_df  = pd.DataFrame(W, index=tool_names, columns=tool_names)
    matrix_path = output_path.replace('.csv', '_cost_matrix.csv')
    matrix_df.to_csv(matrix_path)
    print(f"16×16 cost matrix saved → {matrix_path}")

    # Summary
    successful   = sum(1 for r in all_results if r['success'])
    total_tokens = sum(r['total_tokens'] for r in all_results)
    nonzero      = W[W > 0]
    active_edges = int((W > 0).sum() // 2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Success rate:    {successful}/{n_tasks} ({100*successful/n_tasks:.1f}%)")
    print(f"Total tokens:    {total_tokens:,}")
    print(f"Est. cost:       ~${total_tokens * 0.0000004:.3f}")
    print(f"Active edges:    {active_edges}/120 possible")
    if len(nonzero) > 0:
        print(f"Weight range:    {nonzero.min():.0f}–{nonzero.max():.0f} tokens")
        print(f"CV:              {nonzero.std()/nonzero.mean():.3f}")

    by_c = df.groupby('complexity')['total_tokens'].agg(['mean', 'std'])
    by_c['cv'] = by_c['std'] / by_c['mean']
    print(f"\nTokens by complexity:")
    print(by_c.round(1))

    print(f"\n✓ Done. Use '{matrix_path}' in qaoa_experiment_16_I11.py")
    return df, W


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate 16-node MCP data for QAOA n=16 experiment')
    parser.add_argument('--tasks',  type=int, default=150)
    parser.add_argument('--output', type=str, default='mcp_agent_data_16_I11.csv')
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in your .env file.")
        exit(1)

    run_experiment(args.tasks, args.output)
