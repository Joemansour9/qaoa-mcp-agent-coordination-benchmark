"""
MCP Agent Interaction Data Generator — 16 Node Version — INSTANCE I9
======================================================================
Generates the 16x16 cost matrix for Instance I9: a topology-shifted
MCP tool-interaction graph for patent case P202602120.

Identical pipeline to mcp_data_generator_16.py (same MCP tool
definitions, same GPT-4o-mini agent harness, same execute_tool
simulation, same cost-matrix construction) — the ONLY change is the
TASKS dictionary and task-selection logic below.

Task-mix design for I9 — WEBSEARCH AS HUB (star topology):
  - Every task is a two-step "search web, then use ONE spoke server"
    pattern. Tasks are DETERMINISTICALLY round-robin cycled across
    all 7 other servers (FileSystem, Database, Analytics, Email,
    Calendar, CodeExecution, VectorSearch) as the "spoke", so each
    spoke gets roughly equal representation.
  - No task combines two non-WebSearch servers, so spoke<->spoke
    edges should be rare/absent — WebSearch becomes the highest-degree
    node (a hub connecting to all 7 other servers roughly evenly),
    unlike I1 where multiple servers have comparable degree.

Expected effect: E_I9 should show WebSearch with high degree (7
distinct connections) while spoke servers connect almost exclusively
through WebSearch rather than to each other — a star-topology
divergence pattern, distinct from I6 (isolation) and I7/I8
(pairwise concentration).

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

SPOKES = ["filesystem", "database", "analytics", "email", "calendar", "codeexec", "vectorsearch"]

TASKS = {
    "simple": {
        "filesystem":   ["Search the web for the latest Python version, then write the finding to a file"],
        "database":     ["Search the web for competitor pricing, then query the database to compare against our records"],
        "analytics":    ["Search the web for industry benchmarks, then run analysis comparing them to our data"],
        "email":        ["Search the web for conference dates, then email the team with the details"],
        "calendar":     ["Search the web for the venue's availability, then create a calendar event"],
        "codeexec":     ["Search the web for the latest library version, then run code to test compatibility"],
        "vectorsearch": ["Search the web for related articles, then embed the key findings for future semantic search"],
    },
    "medium": {
        "filesystem":   ["Search the web for best practices on config management, fetch the top article, and write a summary to a config_notes.txt file"],
        "database":     ["Search the web for current data retention regulations, fetch the top result, and query the database for records that may be affected"],
        "analytics":    ["Search the web for two competitor benchmark reports, fetch both, and run an analysis comparing them to our internal metrics"],
        "email":        ["Search the web for upcoming industry conferences, fetch details on the top two, and email the team a comparison with recommendations"],
        "calendar":     ["Search the web for public holidays next quarter, fetch the full list, and create calendar events blocking out each one"],
        "codeexec":     ["Search the web for a known bug in our current library version, fetch the relevant GitHub issue, and run code to verify whether it affects us"],
        "vectorsearch": ["Search the web for recent papers on our research topic, fetch the top three abstracts, and embed them for future semantic search"],
    },
    "complex": {
        "filesystem":   ["Search the web for a full compliance checklist relevant to our industry, fetch the complete document, and write a structured checklist file with each item as a separate section"],
        "database":     ["Search the web for a list of newly announced data-privacy regulations, fetch details on each, and query the database to identify which existing records need review under each regulation"],
        "analytics":    ["Search the web for three independent industry analyst reports on our sector, fetch each one, and run a combined analysis reconciling their differing growth projections against our own data"],
        "email":        ["Search the web for the contact details of five potential partner organizations, fetch their public profiles, and send a personalized outreach email to each based on what was found"],
        "calendar":     ["Search the web for the travel and visa requirements for an upcoming international conference, fetch the relevant government pages, and create a full calendar itinerary of prep deadlines working backward from the conference date"],
        "codeexec":     ["Search the web for the changelog of a major dependency's latest release, fetch the full changelog, and run code to test each breaking change against our current usage"],
        "vectorsearch": ["Search the web for the ten most-cited papers in our research area published this year, fetch each abstract, and embed all ten for a new semantic search index of current literature"],
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

def allocate_spoke_slots(per_complexity: int, spokes: list):
    """
    Deterministically assigns each of the per_complexity task slots in
    a tier to one of the 7 spoke servers via round-robin (i % len(spokes)).
    No RNG involved in WHICH spoke a slot gets — fully reproducible by
    inspection. This gives each spoke roughly per_complexity/7 tasks
    per tier, spreading WebSearch's connections evenly across all 7
    other servers rather than concentrating on any one of them.
    """
    return [spokes[i % len(spokes)] for i in range(per_complexity)]


def run_experiment(n_tasks: int, output_path: str):
    per_complexity = n_tasks // 3
    spoke_assignment = allocate_spoke_slots(per_complexity, SPOKES)
    from collections import Counter
    spoke_counts = Counter(spoke_assignment)

    print(f"\n{'='*60}")
    print(f"MCP 16-Node Agent Data Generation — INSTANCE I9")
    print(f"Servers: 8 × 2 tools = 16 nodes")
    print(f"Tasks:   {n_tasks} ({per_complexity} per complexity level)")
    print(f"Model:   gpt-4o-mini")
    print(f"Est cost: ~${n_tasks * 0.001:.2f}")
    print(f"WebSearch-hub design: every task = WebSearch + 1 spoke, "
          f"round-robin across {len(SPOKES)} spokes")
    print(f"  Per-tier spoke counts (deterministic): {dict(spoke_counts)}")
    print(f"{'='*60}\n")

    all_results = []
    task_id     = 0
    random.seed(602120)  # only affects WHICH prompt variant is picked
                          # within a spoke's pool, never which spoke a
                          # task slot gets (that's fixed by round-robin above)
    spoke_count_by_tier = {}

    for complexity in ['simple', 'medium', 'complex']:
        pools = TASKS[complexity]
        print(f"\n── {complexity.upper()} tasks ({per_complexity}) ──")

        tier_spoke_counts = Counter()
        for i in range(per_complexity):
            spoke = spoke_assignment[i]
            task = random.choice(pools[spoke])
            tier_spoke_counts[spoke] += 1
            task_id += 1
            print(f"  [{task_id:3d}/{n_tasks}] [{spoke[:4].upper():4s}] {task[:50]}...",
                  end=' ', flush=True)
            result = run_agent_task(task, complexity, task_id)
            all_results.append(result)
            status = "✓" if result['success'] else "✗"
            print(f"{status} | tools={result['n_tool_calls']} "
                  f"| tokens={result['total_tokens']}")
            time.sleep(0.3)

        spoke_count_by_tier[complexity] = dict(tier_spoke_counts)
        print(f"  (spoke distribution this tier: {dict(tier_spoke_counts)})")

    print(f"\nOverall spoke distribution: {dict(spoke_counts)} "
          f"(target: even split across {len(SPOKES)} spokes)")

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
    parser.add_argument('--output', type=str, default='mcp_agent_data_16_I9.csv')
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in your .env file.")
        exit(1)

    run_experiment(args.tasks, args.output)