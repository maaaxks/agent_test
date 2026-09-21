import os
import json
from typing import TypedDict, Annotated, Literal, Optional, Dict, Any, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

#model setup
model = ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

#embeddings 
embeddings=HuggingFaceEmbeddings(
    # intfloat/multilingual-e5-small
    model_name="sergeyzh/rubert-tiny-sts-v2", 
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# dirs
REPORT_PATH="./pdf_files/itogi_2025_aiready.pdf"
PERSIST_DIR="./sber_report_index"

def load_and_index_pdf(pdf_path: str):
    """load and index pdf file for chroma"""
    loader=PyPDFLoader(pdf_path)
    pages=loader.load_and_split()
    print(f'loaded {len(pages)} pages')

    splitter =RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks=splitter.split_documents(pages)
    print(f'created {len(chunks)} chunks')

    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    return vectorstore

if os.path.exists(PERSIST_DIR):
    vectorstore= Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
    print("index loaded from disk")
elif os.path.exists(REPORT_PATH):
    vectorstore = load_and_index_pdf(REPORT_PATH)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("index created and saved")
else:
    print(f"report not found at {REPORT_PATH}")
    exit(1)

# state definition
class FinanceAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str
    context: str
    answer: str

# tools
@tool
def summiraze_section(section_name: str)->str:
    """Summarize a specific section of the financial report.

    Args:
        section_name: Name of the section (e.g., 'management report', 'risk factors')
    """

    docs=retriever.invoke(section_name)
    context='\n\n'.join(doc.page_content for doc in docs[:5])
    response=model.invoke([
        SystemMessage(content=(
            "You are a financial analyst. Summarize the following section from the report. "
            "Be concise but comprehensive. Extract key points, numbers, and conclusions.\n\n"
            f"Section: {section_name}"
        )),
        HumanMessage(content=(
            f"Context: {context}, \n provide a structured summary."
        ))
    ])
    return response.content

@tool
def extract_financials(metric: Optional[str]=None)-> Dict[str, Any]:
    """Extract key financial metrics from the report.

    Args:
        metric: Specific metric (e.g., 'revenue', 'net profit', 'ROE'). If None, returns all.
    """
    query="financial results revenue profit ROE NPL EBITDA"
    docs=retriever.invoke(query)
    context='\n\n'.join(doc.page_content for doc in docs[:5])

    response=model.invoke([
        SystemMessage(content=(
            "Extract financial metrics from the context. Return ONLY valid JSON.\n"
            'Format: {"metric_name": "value with unit", ...}\n'
            "Include: revenue, net_profit, roe, npl, assets, equity, credit_portfolio"
        )),
        HumanMessage(content=f"Context:\n{context}\n\nExtract all financial metrics.")
    ])
    try:
        data=json.loads(response.content)
        if metric and metric in data:
            return {metric: data[metric]}
        return data
    except Exception:
        return {"raw_response": response.content}

@tool
def search_in_report(query: str)-> str:
    """Search for specific information within the report.

    Args:
        query: What to find (e.g., 'dividend policy', 'ESG strategy')
    """
    docs=retriever.invoke(query)
    if not docs:
        return "No relevant information found in the report."

    res=[]
    for i, doc in enumerate(docs[:5], 1):
        page=doc.metadata.get('page', 'unknown')
        res.append(f"[Source {i}, Page {page}]\n{doc.page_content}\n")

    return "\n---\n".join(res)

@tool
def get_table(table_name: str)-> str:
    """Extract and structure a specific table from the report.

    Args:
        table_name: Name or description of the table (e.g., 'income statement', 'balance sheet')
    """
    docs=retriever.invoke(f'{table_name} table')
    context='\n\n'.join(doc.page_content for doc in docs[:5])
    response=model.invoke([
        SystemMessage(content=(
            "Extract and structure the table data from the context. "
            "Identify rows and columns. Return in a clear markdown table format. "
            "If table is not found, say so."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nExtract table: {table_name}")
    ])
    return response.content

@tool 
def web_search(query: str)-> str:
    """Search the web for recent news and analysis about Sberbank.

    Args:
        query: Search query about Sberbank (e.g., 'Sberbank financial results 2025')
    """
    try:
        from tavily import TavilyClient
        tavily_key=os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            return "Error: TAVILY_API_KEY not found."
        client=TavilyClient(api_key=tavily_key)
        response=client.search(f"Sberbank {query}", max_results=5)

        results=[]
        for item in response.get("results", []):
            results.append(
                f"Title: {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Content: {item['content'][:400]}\n"
            )
        return "\n---\n".join(results) if results else "No web results found."
    except ImportError:
        return "Error: tavily-python not installed."
    except Exception as e:
        return f"Web search error: {str(e)}"

# nodes
def classify_query_node(state: FinanceAgentState)-> dict:
    last_message=state['messages'][-1].content
    response=model.invoke([
        SystemMessage(content=(
            "Classify the user's query into ONE category.\n"
            "Categories: summary, financials, search, table, web\n\n"
            "Rules:\n"
            "- summary -> user wants a summary of a section\n"
            "- financials -> user asks about numbers, metrics\n"
            "- search -> general question about the report\n"
            "- table -> user asks for a specific table\n"
            "- web -> user asks about news, external info\n\n"
            "Respond with ONLY the category name."
        )),
        HumanMessage(content=(f'query : {last_message}'))
    ])
    category=response.content.strip().lower()
    valid={"summary", "financials", "search", "table", "web"}
    if category not in valid:
        category="search"
    print(f"[Classifier] Category: {category}")
    return {"query_type": category} 

def handle_summary(state:FinanceAgentState)-> dict:
    message=state['messages'][-1].content
    res=summiraze_section.invoke({'section_name': message})
    return {'answer': res, 'messages': [AIMessage(content=res)]}

def handle_financials(state: FinanceAgentState)-> dict:
    message=state['messages'][-1].content
    res= extract_financials.invoke({"metric": None})
    if isinstance(res, dict):
        res=json.dumps(res, indent=2, ensure_ascii=False)
    return {"answer": res, "messages": [AIMessage(content=res)]}

def handle_search(state: FinanceAgentState)-> dict:
    message=state['messages'][-1].content
    res=search_in_report.invoke({'query': message})
    return {"answer": res, "messages": [AIMessage(content=res)]}

def handle_table(state: FinanceAgentState) -> dict:
    message=state["messages"][-1].content
    res=get_table.invoke({"table_name": message})
    return {"answer": res, "messages": [AIMessage(content=res)]}

def handle_web(state: FinanceAgentState) -> dict:
    message=state["messages"][-1].content
    res=web_search.invoke({"query": message})
    return {"answer": res, "messages": [AIMessage(content=res)]}

# routing
def route_by_type(state: FinanceAgentState)-> Literal[
    "handle_summary", "handle_financials", "handle_search", "handle_table", "handle_web"
]:
    query_type = state.get("query_type", "search")
    return f"handle_{query_type}"

# graph
workflow=StateGraph(FinanceAgentState)

workflow.add_node("classify", classify_query_node)
workflow.add_node("handle_summary", handle_summary)
workflow.add_node("handle_financials", handle_financials)
workflow.add_node("handle_search", handle_search)
workflow.add_node("handle_table", handle_table)
workflow.add_node("handle_web", handle_web)

workflow.add_edge(START, "classify")
workflow.add_conditional_edges(
    "classify",
    route_by_type,
    {
        "handle_summary": "handle_summary",
        "handle_financials": "handle_financials",
        "handle_search": "handle_search",
        "handle_table": "handle_table",
        "handle_web": "handle_web",
    }
)
workflow.add_edge("handle_summary", END)
workflow.add_edge("handle_financials", END)
workflow.add_edge("handle_search", END)
workflow.add_edge("handle_table", END)
workflow.add_edge("handle_web", END)

checkpoinetr=MemorySaver()
agent=workflow.compile(checkpointer=checkpoinetr)

# exec
def process_query(message: str, thread_id: str="user_1")->str:
    config={"configurable": {"thread_id": thread_id}}
    print(f"\n{'=' * 60}\nQuery: {message}\n{'=' * 60}")

    result=agent.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "query_type": "",
            "context": "",
            "answer": "",
        },
        config=config
    )
    return result.get("answer", "No answer generated.")


def main():
    print("\nTools: summary | financials | search | table | web")
    print("Type 'exit' to quit.\n")

    thread_id="user_1"
    while True:
        query=input("\nQuestion: ").strip()
        if query.lower()=="exit":
            print("Goodbye!")
            break
        if not query:
            print("Please enter a valid question.")
            continue

        answer=process_query(query, thread_id)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()