"""
MCP Agent Interaction Data Generator — 16 Node Version — INSTANCE I6
======================================================================
Generates the 16x16 cost matrix for Instance I6: a topology-shifted
MCP tool-interaction graph for patent case P202602120.

Identical pipeline to mcp_data_generator_16.py (same MCP tool
definitions, same GPT-4o-mini agent harness, same execute_tool
simulation, same cost-matrix construction) — the ONLY change is the
TASKS dictionary below. This is deliberate: I6's evidentiary standing
depends on it being measured the same way as I1-I5, differing only in
which real task prompts were run.

Task-mix design for I6:
  - ~85-90% of tasks are drawn from FileSystem / WebSearch / Analytics /
    Email / Calendar / CodeExecution / VectorSearch only — Database is
    never mentioned or required, so the agent should not route through
    it during these tasks.
  - ~10-20% of tasks (1 in 10 simple, 1 in 10 medium, 1 in 5 complex)
    are deliberately DATABASE-ISOLATED: they only require
    database_query_db / database_insert_record and no other tool, so
    Database still appears in the graph but as its own disconnected
    component rather than a hub connected to every other server.

Expected effect on the resulting cost matrix: E_I6 should show ~0
Database<->other-server edges (vs. I1, where Database connects densely
to every other server) and a small number of Database<->Database
edges — an edge-SET divergence from I1, not merely a reweighting.

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
  python mcp_data_generator_16.py --tasks 150 --output mcp_agent_data_16.csv
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
    # Server A — FileSystem (nodes 0, 1)
    {
        "type": "function",
        "function": {
            "name": "filesystem_read_file",
            "description": "MCP FileSystem Server: Read a file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem_write_file",
            "description": "MCP FileSystem Server: Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    # Server B — WebSearch (nodes 2, 3)
    {
        "type": "function",
        "function": {
            "name": "websearch_search_web",
            "description": "MCP WebSearch Server: Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "websearch_fetch_url",
            "description": "MCP WebSearch Server: Fetch content from a specific URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    # Server C — Database (nodes 4, 5)
    {
        "type": "function",
        "function": {
            "name": "database_query_db",
            "description": "MCP Database Server: Execute a database query and return results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query"},
                    "database": {"type": "string", "description": "Database name"}
                },
                "required": ["query", "database"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "database_insert_record",
            "description": "MCP Database Server: Insert a record into a database table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name"},
                    "record": {"type": "object", "description": "Record to insert"}
                },
                "required": ["table", "record"]
            }
        }
    },
    # Server D — Analytics (nodes 6, 7)
    {
        "type": "function",
        "function": {
            "name": "analytics_run_analysis",
            "description": "MCP Analytics Server: Run statistical analysis on provided data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Data to analyze (JSON string)"},
                    "analysis_type": {"type": "string", "description": "Type: summary|correlation|trend"}
                },
                "required": ["data", "analysis_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analytics_generate_report",
            "description": "MCP Analytics Server: Generate a formatted report from analysis results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_results": {"type": "string", "description": "Results to report"},
                    "format": {"type": "string", "description": "Output format: markdown|json|text"}
                },
                "required": ["analysis_results", "format"]
            }
        }
    },
    # Server E — Email (nodes 8, 9)
    {
        "type": "function",
        "function": {
            "name": "email_send_email",
            "description": "MCP Email Server: Send an email to specified recipients.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body content"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "email_read_inbox",
            "description": "MCP Email Server: Read emails from inbox with optional filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {"type": "string", "description": "Email folder to read"},
                    "limit": {"type": "integer", "description": "Maximum emails to return"}
                },
                "required": ["folder"]
            }
        }
    },
    # Server F — Calendar (nodes 10, 11)
    {
        "type": "function",
        "function": {
            "name": "calendar_create_event",
            "description": "MCP Calendar Server: Create a new calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "start_time": {"type": "string", "description": "Event start time (ISO format)"},
                    "duration_minutes": {"type": "integer", "description": "Duration in minutes"},
                    "attendees": {"type": "string", "description": "Comma-separated attendee emails"}
                },
                "required": ["title", "start_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_list_events",
            "description": "MCP Calendar Server: List calendar events within a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date (ISO format)"},
                    "end_date": {"type": "string", "description": "End date (ISO format)"}
                },
                "required": ["start_date", "end_date"]
            }
        }
    },
    # Server G — CodeExecution (nodes 12, 13)
    {
        "type": "function",
        "function": {
            "name": "code_run_code",
            "description": "MCP CodeExecution Server: Execute a code snippet and return results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute"},
                    "language": {"type": "string", "description": "Programming language: python|javascript|bash"}
                },
                "required": ["code", "language"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_get_output",
            "description": "MCP CodeExecution Server: Retrieve output from a previous code execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "execution_id": {"type": "string", "description": "Execution ID to retrieve output for"}
                },
                "required": ["execution_id"]
            }
        }
    },
    # Server H — VectorSearch (nodes 14, 15)
    {
        "type": "function",
        "function": {
            "name": "vector_embed_query",
            "description": "MCP VectorSearch Server: Generate embeddings for a text query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to embed"},
                    "model": {"type": "string", "description": "Embedding model to use"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vector_search_index",
            "description": "MCP VectorSearch Server: Search a vector index for similar documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "index_name": {"type": "string", "description": "Vector index to search"},
                    "top_k": {"type": "integer", "description": "Number of results to return"}
                },
                "required": ["query", "index_name"]
            }
        }
    }
]

# Tool name → node index (16 nodes)
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
# 2.  TASK DEFINITIONS — designed to activate all 8 servers
# ═══════════════════════════════════════════════════════════════════════════════

# I6 DESIGN NOTE: Database isolation is controlled explicitly by
# DB_ISOLATION_FRACTION below, sampled per-task at run time — NOT by
# pool-cycling arithmetic. This makes the split exact and easy to state
# in the patent record: 95% of task executions are drawn only from
# non-Database tools; 5% are drawn only from Database tools. No task
# mixes Database with any other server.
DB_ISOLATION_FRACTION = 0.05  # target fraction of tasks that are DB-only

TASKS = {
    "simple": {
        "non_db": [
            "Search the web for the latest Python version, run a quick Python script to print it, and send the result via email to the team",
            "Read the file config.json, embed its key fields using vector search, and save a summary to config_summary.txt",
            "Search for 'quantum computing trends', fetch the top URL, and send a summary email with key findings",
            "Run a Python script to generate a random dataset, analyze it statistically, and write the results to output.txt",
            "Read the inbox for messages about project deadlines, create calendar events for each deadline found",
            "Search the vector index for documents related to 'machine learning', fetch the top result URL, and generate a report",
            "Embed the query 'security vulnerabilities', search the vector index, and send findings via email",
            "Read the file requirements.txt, run code to validate dependencies, and create a calendar event for the review meeting",
            "Search the web for team offsite venues, fetch details from the top result, and schedule a calendar event",
            "Fetch the top result for 'best code editors 2026', run a Python script to log the finding, and read it back from the file",
            "Read inbox for the weekly newsletter, embed its summary for the vector index, and generate a short report",
            "Run code to convert a CSV to JSON, write the output to a file, and email the file location to the team",
        ],
        "isolated": [
            "Query the users database for active accounts and insert a summary record into the audit table in the same database",
        ],
    },
    "medium": {
        "non_db": [
            "Research quantum computing frameworks online, run code to benchmark them, generate an analysis report, and email the summary to the team with a calendar invite for follow-up",
            "Read inbox for customer feedback, embed feedback text for vector search, run sentiment analysis code, generate a report, and schedule a review meeting",
            "Fetch competitor pricing data from the web, run code to compare pricing trends, generate a report, and email stakeholders with meeting invite",
            "Search for security vulnerabilities, read current policy files, embed policy text, search vector index for related incidents, run analysis code, and write updated advisory",
            "Read sales data files, run code to calculate variance, embed the findings, search similar historical patterns in vector index, generate executive report, and email to leadership",
            "Search for ML best practices, fetch paper URLs, run code to extract key metrics, embed findings for future search, generate analysis report, and schedule team review",
            "Read inbox for client requests, embed request text, search vector index for similar past projects, create calendar events for client calls, and generate a proposal report",
            "Fetch API documentation from web, run code to test endpoints, generate technical report, email developer team, and schedule sprint planning meeting",
            "Read configuration files, run code to validate settings, search vector index for known issues, generate audit report, email security team, and schedule remediation meeting",
            "Search the web for conference talks on vector search, fetch two talk summaries, embed them for the index, and email the digest to the team with a calendar reminder",
            "Read a batch of log files, run code to parse error rates, generate a report, embed the report for future retrieval, and schedule a postmortem meeting",
        ],
        "isolated": [
            "Query the transactions database for entries older than 90 days, then insert a summary of the count into the archive_log table in the same database",
        ],
    },
    "complex": {
        "non_db": [
            "Conduct full market analysis: search web for trends, fetch 3 competitor URLs, run code to parse pricing data, embed market data for vector search, run correlation analysis, write detailed report to market_analysis.md, generate executive summary, email to leadership team, and schedule quarterly review meeting",
            "Perform complete system audit: read all config files, search for known CVEs online, fetch vulnerability details, run code to test system exposure, embed audit findings, search vector index for similar past incidents, run statistical analysis, write findings to audit_report.txt, generate executive security report, email CISO, and schedule remediation planning meeting",
            "Build AI pipeline report: read model config files, search arXiv for latest papers, fetch paper abstracts, run code to extract benchmarks, embed abstracts for semantic search, search vector index for related work, run comparative analysis, write technical findings to ml_report.md, generate stakeholder summary, email ML team and leadership, and create calendar events for model review sessions",
            "Perform DevOps pipeline audit: read deployment config files, search web for best practices, fetch documentation URLs, run code to test deployment scripts, embed script content for search, search vector index for known issues, run statistical analysis on deployment success rates, write technical report to devops_audit.md, generate executive summary, email DevOps and engineering leadership, and create calendar events for improvement sprints",
            "Run a full content-ops workflow: read draft files, search the web for citation sources, fetch and embed the source pages, search the vector index for overlapping prior work, run code to check formatting, generate a publication-ready report, email the editor, and schedule a publication calendar event",
        ],
        "isolated": [
            "Perform a full database maintenance pass: query the compliance database for expired records, then query the incidents database for unresolved tickets older than a threshold, and insert consolidated cleanup and status records back into the database tables",
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

def allocate_isolated_slots(n_tasks: int, per_complexity: int, fraction: float):
    """
    Deterministically decide exactly how many tasks (and which task
    slots) are DB-isolated — no RNG involved in this decision, so the
    count is exact and reproducible by inspection of this function alone.

    n_tasks=150 is not divisible by 20, so an exact 5.00% (1-in-20)
    split is not representable as a whole number of tasks. We take the
    nearest whole-task count via standard rounding and report the exact
    resulting percentage rather than rounding the percentage itself.
    Distributed across the three complexity tiers via the largest-
    remainder method (deterministic, order-independent).
    """
    tiers = ["simple", "medium", "complex"]
    total_isolated = round(n_tasks * fraction)

    base = total_isolated // len(tiers)
    remainder = total_isolated - base * len(tiers)
    counts = {t: base for t in tiers}
    for t in tiers[:remainder]:          # fixed, deterministic tie-break order
        counts[t] += 1

    # Deterministic, evenly-spaced slot positions within each tier's
    # `per_complexity` tasks (not random placement).
    slot_positions = {}
    for t in tiers:
        k = counts[t]
        slot_positions[t] = sorted(
            int((j + 0.5) * per_complexity / k) for j in range(k)
        ) if k > 0 else []

    return counts, slot_positions, total_isolated


def run_experiment(n_tasks: int, output_path: str):
    per_complexity = n_tasks // 3
    isolated_counts, isolated_slots, total_isolated = allocate_isolated_slots(
        n_tasks, per_complexity, DB_ISOLATION_FRACTION
    )
    actual_pct = 100 * total_isolated / n_tasks

    print(f"\n{'='*60}")
    print(f"MCP 16-Node Agent Data Generation — INSTANCE I6")
    print(f"Servers: 8 × 2 tools = 16 nodes")
    print(f"Tasks:   {n_tasks} ({per_complexity} per complexity level)")
    print(f"Model:   gpt-4o-mini")
    print(f"Est cost: ~${n_tasks * 0.001:.2f}")
    print(f"DB-isolation target: {100*DB_ISOLATION_FRACTION:.1f}%  "
          f"→ deterministic allocation: {total_isolated}/{n_tasks} tasks "
          f"({actual_pct:.2f}% exact)")
    print(f"  Per-tier counts: {isolated_counts}")
    print(f"  Per-tier slot positions (0-indexed): {isolated_slots}")
    print(f"{'='*60}\n")

    all_results = []
    task_id     = 0
    random.seed(602120)  # tie to patent case number; only affects WHICH
                          # prompt variant is picked within a pool, never
                          # whether a task is DB-isolated (that's fixed above)
    isolated_count_by_tier = {}

    for complexity in ['simple', 'medium', 'complex']:
        pools = TASKS[complexity]
        non_db_pool, isolated_pool = pools["non_db"], pools["isolated"]
        isolated_idx_set = set(isolated_slots[complexity])
        print(f"\n── {complexity.upper()} tasks ({per_complexity}) ──")

        n_isolated_this_tier = 0
        for i in range(per_complexity):
            is_isolated = i in isolated_idx_set
            if is_isolated:
                task = random.choice(isolated_pool)
                n_isolated_this_tier += 1
            else:
                task = random.choice(non_db_pool)
            task_id += 1
            print(f"  [{task_id:3d}/{n_tasks}] {'[DB]' if is_isolated else '    '} {task[:55]}...",
                  end=' ', flush=True)
            result = run_agent_task(task, complexity, task_id)
            all_results.append(result)
            status = "✓" if result['success'] else "✗"
            print(f"{status} | tools={result['n_tool_calls']} "
                  f"| tokens={result['total_tokens']}")
            time.sleep(0.3)

        isolated_count_by_tier[complexity] = n_isolated_this_tier
        print(f"  ({n_isolated_this_tier}/{per_complexity} = "
              f"{100*n_isolated_this_tier/per_complexity:.1f}% DB-isolated this tier)")

    total_isolated = sum(isolated_count_by_tier.values())
    print(f"\nOverall DB-isolated tasks: {total_isolated}/{n_tasks} "
          f"({100*total_isolated/n_tasks:.1f}%, target {100*DB_ISOLATION_FRACTION:.0f}%)")

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
    print(f"Weight range:    {nonzero.min():.0f}–{nonzero.max():.0f} tokens")
    print(f"CV:              {nonzero.std()/nonzero.mean():.3f}")

    by_c = df.groupby('complexity')['total_tokens'].agg(['mean', 'std'])
    by_c['cv'] = by_c['std'] / by_c['mean']
    print(f"\nTokens by complexity:")
    print(by_c.round(1))

    print(f"\n✓ Done. Use '{matrix_path}' in qaoa_experiment_16.py")
    return df, W


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate 16-node MCP data for QAOA n=16 experiment')
    parser.add_argument('--tasks',  type=int, default=150)
    parser.add_argument('--output', type=str, default='mcp_agent_data_16_I6.csv')
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in your .env file.")
        exit(1)

    run_experiment(args.tasks, args.output)