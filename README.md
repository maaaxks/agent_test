# AI Agents with LangGraph

A collection of three AI agent implementations built with **LangChain**, **LangGraph** and **Groq**.

The project demonstrates several common agent architectures:

1. **Research Agent** — searches the web and generates research reports with human approval.
2. **Financial Report Agent** — uses RAG to answer questions about a financial PDF(Sber 2025) and can additionally search the web.
3. **Supervisor Agent** — routes user requests between a SQL agent and a data visualization agent (chinook db).

The project is primarily intended as a practical exploration of agentic workflows, tool calling, RAG, routing, memory and multi-agent architectures.

---

## Architecture Overview

```text
                        
      ┌──────────────┐      ┌──────────────┐      ┌─────────────────┐
      │ Research     │      │ Financial    │      │ Supervisor      │
      │ Agent        │      │ Agent        │      │ Agent           │
      └──────┬───────┘      └──────┬───────┘      └────────┬────────┘
             │                     │                        │
             ▼                     ▼                ┌───────┴────────┐
        Tavily Search         PDF / Chroma          │                │
             │                Vector Store          ▼                ▼
             ▼                                      SQL Agent    Analytics Agent
        Report Draft                                  │                │
             │                                        ▼                ▼
             ▼                                      SQLite          Charts
      Human Review
             │
       ┌─────┴─────┐
       │           │
     Approve     Revise
       │           │
       ▼           └──────────────► Draft
   Save Report
```

---

# Agents

## 1. Research Agent

**File:** `agent.py`

The first agent is a research assistant that takes a topic, searches the web, generates a structured research report and asks the user to approve it before saving.

### Workflow

```text
User enters topic
       │
       ▼
Web Search
(Tavily)
       │
       ▼
Search Results
       │
       ▼
LLM Draft Generation
       │
       ▼
Human Review
       │
   ┌───┴────┐
   │        │
 Approve   Revise
   │        │
   ▼        ▼
 Save      LLM
 Report    Revision
            │
            └──────► Human Review
```

### Components

**LLM**

The agent uses Groq through LangChain:

```python
ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
```

### Tools

#### `web_search`

A Tavily-powered search tool.

It:

* accepts a search query;
* performs a web search;
* retrieves up to 5 results;
* extracts title, URL and a short content fragment;
* returns the collected information to the agent.

### LangGraph nodes

The workflow contains five main nodes:

* `research` — performs the web search;
* `draft` — generates the research report;
* `review` — asks the user whether the report should be approved or revised;
* `revise` — regenerates the report using user feedback;
* `finalize` — saves the approved report as a Markdown file.

The resulting graph contains a conditional branch after the review step:

```text
research → draft → review
                    │
             ┌──────┴──────┐
             │             │
          approve        revise
             │             │
             ▼             ▼
          finalize ←──── revise
```

### Output

Approved reports are saved into:

```text
reports/
```

with filenames generated from the research topic:

```text
report_<topic>.md
```

---

## 2. Financial Report Agent

**File:** `agent_2.py`

The second agent is a financial research assistant specialized in answering questions about a financial report.

Unlike the first agent, it does not search the entire web for every question. Instead, it uses **RAG (Retrieval-Augmented Generation)** over a PDF document and can use web search when the question requires external or recent information.

### Architecture

```text
                       User Query
                           │
                           ▼
                    Query Classifier
                           │
       ┌───────────┬──────┼──────┬───────────┐
       ▼           ▼      ▼      ▼           ▼
    Summary    Financials Search Table       Web
       │           │      │      │           │
       ▼           ▼      ▼      ▼           ▼
     RAG         RAG     RAG    RAG       Tavily
       │           │      │      │           │
       └───────────┴──────┴──────┴───────────┘
                           │
                           ▼
                         Answer
```

### RAG Pipeline

The financial report is processed using:

* `PyPDFLoader` — loads the PDF;
* `RecursiveCharacterTextSplitter` — splits the document into chunks;
* `HuggingFaceEmbeddings` — creates embeddings;
* `Chroma` — stores and retrieves document vectors.

The current embedding model is:

```text
sergeyzh/rubert-tiny-sts-v2
```

The document is split using:

```text
chunk_size: 1500
chunk_overlap: 200
```

The vector database is persisted locally in:

```text
sber_report_index/
```

If the index already exists, it is loaded from disk. Otherwise, the PDF is processed and indexed automatically.

### Available Tools

#### `summiraze_section`

Retrieves relevant parts of the report and asks the LLM to produce a structured summary.

Useful for questions such as:

```text
Summarize the management report.
```

#### `extract_financials`

Extracts financial metrics from the report and returns them as JSON.

The tool is designed to work with metrics such as:

* revenue;
* net profit;
* ROE;
* NPL;
* assets;
* equity;
* credit portfolio.

#### `search_in_report`

Performs semantic search over the report and returns relevant passages together with page numbers.

#### `get_table`

Searches for a requested table and asks the LLM to reconstruct it as a Markdown table.

#### `web_search`

Uses Tavily to search for external information and recent news related to Sberbank.

### Query Routing

Before executing a request, the agent classifies it into one of five categories:

```text
summary
financials
search
table
web
```

The classifier then routes the request to the corresponding handler.

For example:

```text
"Summarize the risk factors"
        ↓
     summary
        ↓
summiraze_section()
```

or:

```text
"What is the ROE?"
        ↓
    financials
        ↓
extract_financials()
```

### Memory

The agent uses LangGraph's `MemorySaver` checkpointer.

A `thread_id` is used to maintain state between interactions:

```python
thread_id="user_1"
```

---

## 3. Supervisor / SQL / Analytics Agent

**File:** `agent_3.py`

The third implementation demonstrates a **multi-agent architecture**.

It contains three logical agents:

```text
                    Supervisor Agent
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
          SQL Agent             Analytics Agent
              │                       │
              ▼                       ▼
         SQLite DB                 Charts
```

The supervisor decides which worker should handle the user's request.

### Supervisor Agent

The supervisor receives the user's query and classifies it into:

```text
sql_agent
analytics_agent
```

For example:

```text
"How many customers are from Canada?"
                ↓
           sql_agent
```

while:

```text
"Show revenue by month"
                ↓
        analytics_agent
```

The supervisor does not perform the database analysis itself. Its main responsibility is **routing**.

---

### SQL Agent

The SQL agent works with the SQLite **Chinook** database.

It is created using LangChain's:

```python
SQLDatabaseToolkit
```

The toolkit provides database-specific tools for:

* listing tables;
* inspecting schemas;
* checking SQL queries;
* executing SQL queries.

The agent is explicitly instructed to use a safe read-only workflow:

```text
1. List available tables
2. Inspect relevant schemas
3. Generate SQL
4. Check the SQL query
5. Execute the query
6. Return the result in natural language
```

The system prompt also restricts the agent to read-only operations:

```text
SELECT
```

and prohibits:

```text
INSERT
UPDATE
DELETE
DROP
```

This makes the SQL agent suitable for analytical queries without intentionally modifying the database.

---

### Analytics Agent

The analytics agent is responsible for turning query data into visualizations.

It uses a custom tool:

```python
generate_chart()
```

The tool supports three chart types:

```text
bar
line
pie
```

The intended use cases are:

| Chart  | Example                     |
| ------ | --------------------------- |
| `line` | Revenue over time           |
| `bar`  | Sales by genre              |
| `pie`  | Distribution or proportions |

Charts are generated with Matplotlib and saved into:

```text
charts/
```

The generated filename is based on the chart title and chart type.

Example:

```text
charts/Sales_by_Genre_bar.png
```

---

## Technologies

The project uses the following technologies:

* **Python**
* **LangChain**
* **LangGraph**
* **Groq**
* **Tavily**
* **Chroma**
* **Hugging Face Embeddings**
* **PyPDF**
* **SQLite**
* **Matplotlib**

### Main concepts demonstrated

* LLM tool calling
* LangGraph state machines
* Conditional routing
* Human-in-the-loop workflows
* RAG
* Vector databases
* Semantic search
* Agent memory
* SQL agents
* Multi-agent systems
* Data visualization
* Web search

---

# Project Structure

```text
agent_test/
│
├── agent.py
│   └── Research Agent
│
├── agent_2.py
│   └── Financial Report Agent
│
├── agent_3.py
│   └── Supervisor + SQL + Analytics Agents
│
├── data/
│   └── chinook.db
│
├── pdf_files/
│   └── financial report PDF
│
├── sber_report_index/
│   └── Chroma vector database
│
├── reports/
│   └── generated research reports
│
├── charts/
│   └── generated charts
│
└── README.md
```

---

# Setup

## 1. Clone the repository

```bash
git clone https://github.com/maaaxks/agent_test.git
cd agent_test
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## 3. Install dependencies

Install the required LangChain, LangGraph, Groq, Tavily, Chroma, PDF and visualization packages.

## 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

`TAVILY_API_KEY` is required for agents that use web search.

---

# Running the Agents

## Research Agent

```bash
python agent.py
```

Enter a research topic:

```text
Research topic: Artificial Intelligence in Finance
```

The agent searches the web, creates a draft and asks for approval.

Possible responses:

```text
yes
no
revise
```

For `revise`, additional feedback can be provided.

---

## Financial Report Agent

```bash
python agent_2.py
```

The agent can answer questions about the indexed financial report.

Examples:

```text
What is the company's revenue?
```

```text
Summarize the management report.
```

```text
Find information about dividend policy.
```

```text
Show the income statement.
```

```text
What are the latest news about Sberbank?
```

---

## Supervisor Agent

```bash
python agent_3.py
```

Example questions:

```text
How many customers are from Canada?
```

The supervisor routes the request to the SQL agent.

Another example:

```text
Show sales by genre as a chart.
```

The supervisor routes the request to the analytics agent.

---

# Agent Comparison

| Agent                  | Main Purpose                        | Tools                      | Architecture                   |
| ---------------------- | ----------------------------------- | -------------------------- | ------------------------------ |
| Research Agent         | Web research and report generation  | Tavily                     | Sequential + Human-in-the-loop |
| Financial Report Agent | Questions about a financial PDF     | Chroma, embeddings, Tavily | RAG + routing                  |
| Supervisor Agent       | Database analysis and visualization | SQL Toolkit, Matplotlib    | Multi-agent                    |

---

# What This Project Demonstrates

The project progresses from a relatively simple tool-using agent to more complex architectures:

```text
Agent 1
Tool Calling
    │
    ▼
Research + Human Review
    │
    ▼
Agent 2
RAG + Multiple Tools
    │
    ▼
Agent 3
Supervisor + Multiple Agents
```

This makes the repository a small collection of practical examples of how agent architectures can evolve from a single workflow into a multi-agent system.
