"""
MCP Agent Interaction Data Generator — 16 Node Version — INSTANCE I10
======================================================================
Generates the 16x16 cost matrix for Instance I10: a two-competing-hubs
MCP tool-interaction graph, extending the I6-I9 topology-shift study.

Identical pipeline to mcp_data_generator_16.py (same MCP tool
definitions, same GPT-4o-mini agent harness, same execute_tool
simulation, same cost-matrix construction) — the ONLY change is the
TASKS dictionary below.

Task-mix design for I10 — TWO COMPETING HUBS:
  - A DETERMINISTIC ~40% of tasks are drawn from a "hub_a" pool: they
    touch ONLY filesystem_read_file / filesystem_write_file /
    analytics_run_analysis / analytics_generate_report. This
    concentrates a large fraction of edge weight onto the
    FileSystem<->Analytics pair.
  - A DETERMINISTIC ~40% of tasks are drawn from a "hub_b" pool: they
    touch ONLY database_query_db / database_insert_record /
    calendar_create_event / calendar_list_events. This concentrates a
    comparably large fraction of edge weight onto the
    Database<->Calendar pair.
  - A DETERMINISTIC ~20% ("diverse") minority spreads across the
    remaining 4 servers (WebSearch, Email, CodeExecution,
    VectorSearch) and deliberately excludes all four hub servers
    (FileSystem, Analytics, Database, Calendar) entirely.

Unlike I7/I8 (a single ~70% dominant pair, which lets QAOA solve the
whole Max-Cut instance almost trivially by isolating that one edge),
I10 gives two comparably-weighted dominant pairs with no diverse-pool
bridge between them. The optimizer must actually trade off which pair
to cut, rather than solving the graph by isolating a single edge —
intended to reduce the ceiling-effect risk (r_sim=1.000 at low p) seen
in several of the I6-I9 cells.

Expected effect on the resulting cost matrix: E_I10 should show TWO
comparably high-weight edges (FileSystem<->Analytics and
Database<->Calendar), each roughly isolated from the rest of the
graph, with a low-weight diverse periphery connecting the remaining 4
servers — a "two competing hubs" divergence pattern, distinct from
I6 (isolation), I7/I8 (single pairwise concentration), and I9 (star).

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
  python mcp_data_generator_16_I10.py --tasks 150 --output mcp_agent_data_16_I10.csv
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


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TASK DEFINITIONS — TWO COMPETING HUBS
# ═══════════════════════════════════════════════════════════════════════════════

# I10 DESIGN NOTE: pool assignment per task slot is a fixed, deterministic
# cyclic pattern (not sampled at run time), same mechanism as I6/I7/I8/I9 —
# hub_a, hub_a, hub_b, hub_b, diverse repeating gives an exact 40/40/20 split
# whenever per_complexity is divisible by 5 (50 -> 10 repeats, exact).
HUB_PATTERN = ["hub_a", "hub_a", "hub_b", "hub_b", "diverse"]

TASKS = {
    "simple": {
        "hub_a": [
            "Read the file metrics.csv and run a statistical analysis on its contents",
            "Run an analysis of last week's log file and write the summary back to disk",
            "Read config.json and generate a formatted report of its settings",
            "Run a trend analysis on sales_data.txt and save the output to a new file",
        ],
        "hub_b": [
            "Query the inventory database for low-stock items and create a calendar event to reorder",
            "List this week's calendar events and query the database for attendee records",
            "Create a calendar event for the maintenance window and insert a record noting it in the database",
            "Query the customers database for upcoming renewals and list the relevant calendar events",
        ],
        "diverse": [
            "Search the web for the latest Python version and run a quick script to print it",
            "Send a follow-up email summarizing yesterday's standup",
            "Embed the query 'security vulnerabilities' and search the vector index",
            "Fetch the top result for 'best code editors 2026' and run a Python script to log the finding",
        ],
    },
    "medium": {
        "hub_a": [
            "Read the quarterly metrics file, run a full statistical analysis, and generate an executive report from the results",
            "Read a batch of log files, run code-free trend analysis across them, and write a consolidated report",
            "Read the config history file, run an analysis of setting changes over time, and generate a change report",
            "Read the raw survey data file, run a correlation analysis, and generate a formatted summary report",
        ],
        "hub_b": [
            "Query the transactions database for entries older than 90 days, create calendar events for the review of each batch, and list existing calendar conflicts",
            "List next week's calendar events, query the database for related project records, and create a follow-up calendar event",
            "Query the compliance database for expiring certifications, create calendar reminders for each renewal deadline, and list all reminders created",
            "Create calendar events for the on-call rotation, query the incidents database for prior coverage gaps, and list the updated schedule",
        ],
        "diverse": [
            "Search the web for competitor pricing, fetch the top result, and send a summary email to the team",
            "Run code to parse a CSV export, execute a validation script, and send an email with the results",
            "Search for ML best practices, fetch two paper URLs, and embed the findings for the vector index",
            "Fetch API documentation from the web, run code to test an endpoint, and send an email to the developer team",
        ],
    },
    "complex": {
        "hub_a": [
            "Read all quarterly financial files, run a full statistical analysis reconciling them, generate a detailed executive report, and read back the report file to verify formatting",
            "Read every config file in the deployment set, run a comprehensive drift analysis across them, generate a technical report, and read the final report file to confirm structure",
            "Read the full year of sales files, run a multi-part trend and variance analysis, generate a board-level report, and write a condensed one-page version to a second file",
        ],
        "hub_b": [
            "Query the compliance database for all expired records, query the incidents database separately for unresolved tickets, create calendar events for each remediation deadline, and list the finalized remediation calendar",
            "Query the customer database for accounts up for renewal, create calendar events for each renewal call, list all created events, and query the database again to insert a note that outreach was scheduled",
            "Query the audit database for flagged transactions, create calendar events for each required review meeting, list the resulting review calendar, and insert a summary record back into the audit database",
        ],
        "diverse": [
            "Search the web for a full compliance checklist, fetch the complete document, run code to convert it to a structured format, and send an email with the checklist attached",
            "Search the web for the ten most-cited papers in our research area, fetch each abstract, and embed all ten for a new semantic search index",
            "Run a full content workflow: search the web for citation sources, fetch and embed the source pages, and send the compiled digest via email",
        ],
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

def allocate_hub_slots(per_complexity: int, pattern: list):
    """
    Deterministically assigns each of the per_complexity task slots in a
    tier to a pool ('hub_a' / 'hub_b' / 'diverse') via a fixed repeating
    pattern (i % len(pattern)) — no RNG involved in WHICH pool a slot
    gets, fully reproducible by inspection. With pattern
    [hub_a, hub_a, hub_b, hub_b, diverse] this gives an exact 40/40/20
    split whenever per_complexity is divisible by 5.
    """
    return [pattern[i % len(pattern)] for i in range(per_complexity)]


def run_experiment(n_tasks: int, output_path: str):
    per_complexity = n_tasks // 3
    pool_assignment = allocate_hub_slots(per_complexity, HUB_PATTERN)
    from collections import Counter
    pool_counts = Counter(pool_assignment)

    print(f"\n{'='*60}")
    print(f"MCP 16-Node Agent Data Generation — INSTANCE I10")
    print(f"Servers: 8 × 2 tools = 16 nodes")
    print(f"Tasks:   {n_tasks} ({per_complexity} per complexity level)")
    print(f"Model:   gpt-4o-mini")
    print(f"Est cost: ~${n_tasks * 0.001:.2f}")
    print(f"Two-hub design: FileSystem<->Analytics (hub_a) + "
          f"Database<->Calendar (hub_b), pattern {HUB_PATTERN}")
    print(f"  Per-tier pool counts (deterministic): {dict(pool_counts)}")
    print(f"{'='*60}\n")

    all_results = []
    task_id     = 0
    random.seed(602120)  # only affects WHICH prompt variant is picked
                          # within a pool, never which pool a task slot
                          # gets (that's fixed by the pattern above)
    pool_count_by_tier = {}

    for complexity in ['simple', 'medium', 'complex']:
        pools = TASKS[complexity]
        print(f"\n── {complexity.upper()} tasks ({per_complexity}) ──")

        tier_pool_counts = Counter()
        for i in range(per_complexity):
            pool = pool_assignment[i]
            task = random.choice(pools[pool])
            tier_pool_counts[pool] += 1
            task_id += 1
            tag = {"hub_a": "[HA]", "hub_b": "[HB]", "diverse": "    "}[pool]
            print(f"  [{task_id:3d}/{n_tasks}] {tag} {task[:55]}...",
                  end=' ', flush=True)
            result = run_agent_task(task, complexity, task_id)
            all_results.append(result)
            status = "✓" if result['success'] else "✗"
            print(f"{status} | tools={result['n_tool_calls']} "
                  f"| tokens={result['total_tokens']}")
            time.sleep(0.3)

        pool_count_by_tier[complexity] = dict(tier_pool_counts)
        print(f"  (pool distribution this tier: {dict(tier_pool_counts)})")

    print(f"\nOverall pool distribution: {dict(pool_counts)} "
          f"(target: 40% hub_a / 40% hub_b / 20% diverse)")

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

    print(f"\n✓ Done. Use '{matrix_path}' in qaoa_experiment_16_I10.py")
    return df, W


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate 16-node MCP data for QAOA n=16 experiment')
    parser.add_argument('--tasks',  type=int, default=150)
    parser.add_argument('--output', type=str, default='mcp_agent_data_16_I10.csv')
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in your .env file.")
        exit(1)

    run_experiment(args.tasks, args.output)
