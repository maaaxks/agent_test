import os
import sqlite3
from typing import TypedDict, Annotated, Literal, Optional, Dict, Any, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

import matplotlib
matplotlib.use("Agg")  # non-GUI backend for server-side rendering
import matplotlib.pyplot as plt

load_dotenv()

# model setup
model=ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# db
DB_PATH='./data/chinook.db'
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
print(f"Database loaded: {DB_PATH}")
print(f"Tables: {db.get_usable_table_names()}")

# sql agent
sql_toolkit=SQLDatabaseToolkit(db=db, llm=model)
sql_tools=sql_toolkit.get_tools()

sql_agent=create_agent(
    model=model,
    tools=sql_tools,
    system_prompt=(
        "You are a SQL expert working with the Chinook database (a digital music store).\n\n"
        "Your workflow:\n"
        "1. First, list all tables to understand what's available (sql_db_list_tables)\n"
        "2. Then inspect the schema of relevant tables (sql_db_schema)\n"
        "3. Write a syntactically correct SQLite query\n"
        "4. Double-check your query with sql_db_query_checker\n"
        "5. Execute it with sql_db_query\n"
        "6. Return the final answer to the user in natural language\n\n"
        "Rules:\n"
        "- NEVER write data (INSERT, UPDATE, DELETE, DROP). Only SELECT.\n"
        "- If the query returns an error, fix it and try again.\n"
        "- If no data is found, say so honestly."
    )
)

# analytics agent
# tool fo charts
@tool
def generate_chart(data:str, chart_type: str, title: str = "Data Analysis") -> str:
    """Generate a chart from SQL query results.

    Args:
        data: Formatted result text from SQL query (e.g., rows with labels and values)
        chart_type: Type of chart: bar, line, or pie
        title: Title for the chart
    """
    try:
        lines = [line.strip() for line in data.strip().split("\n") if line.strip()]
        parsed = []

        for line in lines:
            # try different formats of input data
            if "," in line:
                parts = line.split(",", 1)
            elif ":" in line:
                parts = line.split(":", 1)
            elif "(" in line and ")" in line:
                # handle tuple-like output from sql agent
                inner = line.strip("()")
                parts = [p.strip().strip("'\"") for p in inner.split(",")]
                if len(parts) < 2:
                    continue
            else:
                continue

            label = parts[0].strip().strip("'\"")
            value_str = parts[1].strip().strip("'\"")
            try:
                value = float(value_str)
            except ValueError:
                continue
            parsed.append((label, value))

        if not parsed:
            return "Error: Could not parse data for chart. Provide data in format 'label, value' per line."

        labels = [p[0] for p in parsed]
        values = [p[1] for p in parsed]

        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            ax.bar(labels, values, color='steelblue')
            ax.set_ylabel("Value")
        elif chart_type == "line":
            ax.plot(labels, values, marker='o', color='steelblue')
            ax.set_ylabel("Value")
        elif chart_type == "pie":
            ax.pie(values, labels=labels, autopct='%1.1f%%')
        else:
            return f"Error: Unsupported chart type '{chart_type}'. Use bar, line, or pie."

        ax.set_title(title)
        if chart_type != "pie":
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # save chart
        os.makedirs("./charts", exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in "_-" else "_" for c in title)[:50]
        filepath = f"./charts/{safe_title}_{chart_type}.png"
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

        return f"Chart saved: {filepath}"
    except Exception as e:
        return(f"CHart generation error: {str(e)}")

# analytics agent
analytics_agent=create_agent(
    model=model,
    tools=[generate_chart],
    system_prompt=(
        "You are a data visualization specialist. You receive data from SQL queries "
        "and create charts from them.\n\n"
        "Your workflow:\n"
        "1. If you receive raw data, parse it into label-value pairs\n"
        "2. Call generate_chart with the data and appropriate chart type\n"
        "3. Return the path to the saved chart\n\n"
        "Chart type guidance:\n"
        "- Use 'line' for time series (revenue over months, trends over years)\n"
        "- Use 'bar' for comparisons (sales by genre, top artists)\n"
        "- Use 'pie' for proportions (market share, distribution)\n\n"
        "When you receive a question, first get the data from the user or context, "
        "then generate the chart."
    )
)

# state
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    answer: str

# supervisor node
def supervisor_node(state: SupervisorState) -> dict:
    """Decide which agent should handle the query."""
    last_message=state["messages"][-1].content
    response=model.invoke([
        SystemMessage(content=(
            "You are a supervisor managing two agents:\n\n"
            "1. 'sql_agent' — answers questions that require querying the Chinook database "
            "(counts, averages, lists, comparisons, statistics, anything about artists, "
            "albums, tracks, customers, invoices, employees, playlists, sales).\n"
            "Use this for questions that expect a TEXT answer with numbers.\n\n"
            "2. 'analytics_agent' — creates CHARTS and VISUALIZATIONS from database data. "
            "Use this when the user asks to 'show', 'plot', 'chart', 'visualize', or "
            "'graph' data over time or comparisons.\n\n"
            "Respond with ONLY one word: 'sql_agent' or 'analytics_agent'.\n\n"
            "Examples:\n"
            "- 'How many customers are from Canada?' -> sql_agent\n"
            "- 'What is the most expensive track?' -> sql_agent\n"
            "- 'Show revenue by month for 2022' -> analytics_agent\n"
            "- 'Plot top 10 artists by sales' -> analytics_agent\n"
            "- 'Chart sales by genre' -> analytics_agent"
        )),
        HumanMessage(content=(f"query : {last_message}"))
    ])

    next_agent=response.content.strip().lower().replace('"', '').replace("'", "")
    if "analytics" in next_agent:
        next_agent = "analytics_agent"
    else:
        next_agent = "sql_agent"

    print(f"[Supervisor] Routing to: {next_agent}")

    return {"next_agent": next_agent}

# workers agents nodes
def sql_node(state: SupervisorState) -> dict:
    """Run the SQL agent"""
    result=sql_agent.invoke({"messages": state["messages"]})
    answer=result["messages"][-1].content
    print(f"[SQL Agent] Answer ready ({len(answer)} chars)")
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)]
    }

def analytics_node(state: SupervisorState)-> dict:
    result=analytics_agent.invoke({"messages": state['messages']})
    answer=result["messages"][-1].content
    print(f"[Analytics Agent] Answer ready ({len(answer)} chars)")
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)]
    }

# routing
def route_by_agent(state: SupervisorState)-> Literal["sql_agent", "analytics_agent"]:
    """Route to the appropriate worker based on supervisor's decision."""
    return state.get("next_agent", "sql_agent")

# building graph
workflow=StateGraph(SupervisorState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("sql_agent", sql_node)
workflow.add_node("analytics_agent", analytics_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_by_agent,
    {
        "sql_agent": "sql_agent",
        "analytics_agent": "analytics_agent",
    }
)

workflow.add_edge("sql_agent", END)
workflow.add_edge("analytics_agent", END)

checkpointer=MemorySaver()
agent=workflow.compile(checkpointer=checkpointer)

# exec
def process_query(message: str, thread_id: str = "user_1") -> str:
    """Process a user query through the supervisor."""
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'=' * 60}")
    print(f"Query: {message}")
    print('=' * 60)

    result = agent.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "next_agent": "",
            "answer": "",
        },
        config=config
    )
    return result.get("answer", "No answer generated.")


def main():
    print("=" * 60)
    print("SUPERVISOR AGENT WITH SQL AND ANALYTICS WORKERS")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print("\nThe supervisor routes queries to:")
    print("  - sql_agent       : questions about the Chinook database (text answers)")
    print("  - analytics_agent : requests for charts and visualizations")
    print("\nExamples:")
    print("  'How many customers are from Canada?' -> sql_agent")
    print("  'Show revenue by month for 2022' -> analytics_agent")
    print("\nType 'exit' to quit.\n")

    thread_id = "user_1"
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            print("Please enter a valid question.")
            continue

        answer = process_query(query, thread_id)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()