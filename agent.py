import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


#model setup
model = ChatGroq(
    model='openai/gpt-oss-20b',  # Working model
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# state
class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    search_res: str
    draft_report: str
    approved: bool
    final_report: str

@tool
def web_search(query:str) -> str:
    """Search the web for information on a topic.

    Args:
        query: The search query.
    """
    try:
        from tavily import TavilyClient
        client=TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
        response=client.search(query, max_results=5)
        results=[]
        for item in response.get('results', []):
            results.append(
                f"Title: {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Content: {item['content'][:500]}\n"
            )
        return "\n---\n".join(results) if results else "No results found." 
    except Exception as e:
        return f"Search error: str{e}"

# nodes
def research_node(state: ResearchState) -> dict:
    """Search for information on the topic."""
    topic=state['topic']
    print(f'Research for topic {topic}')
    results = web_search.invoke({'query' : topic})
    return {'search_res': results}

def draft_node(state: ResearchState) -> dict:
    """Draft a research report from the search results."""
    topic=state['topic']
    search_res=state['search_res']
    response=model.invoke([
        SystemMessage(content=(
            "You are a research analyst. Write a concise research report based on "
            "the provided search results. Include:\n"
            "1. Executive Summary (2-3 sentences)\n"
            "2. Key Findings (bullet points)\n"
            "3. Analysis (1-2 paragraphs)\n"
            "4. Sources (list URLs from the search results)\n\n"
            f"Topic: {topic}\n\n"
            f"Search Results:\n{search_res}"
        )),
        HumanMessage(f'Write a research report on topic: {topic}')
    ])
    draft=response.content
    return {'draft_report': draft}

def review_node(state: ResearchState) -> dict:
    """Present the draft for human review."""
    print("\n" + "=" * 60)
    print("DRAFT RESEARCH REPORT")
    print("=" * 60)
    print(state["draft_report"])
    print("=" * 60)

    approval=input(f"Approve this report? (yes/no/revise)").strip().lower()
    if approval=='yes':
        return {'approved': True}
    elif approval=='revise':
        feedback=input('What should be changed?')
        return {
            'approved': False,
            'messages': [HumanMessage(content=f'revise report: {feedback}')]
        }
    else:
        return {'approved': False}

def finalize_node(state: ResearchState) -> dict:
    """Save the approved report."""
    report=state["draft_report"]
    topic_slug=state["topic"].lower().replace(" ", "_")[:50]
    filename=f"report_{topic_slug}.md"

    os.makedirs("reports", exist_ok=True)
    filepath=os.path.join("reports", filename)
    with open(filepath, "w") as f:
        f.write(report)

    print(f"\n[Finalize] Report saved to {filepath}")
    return {
        "final_report": filepath,
        "messages": [AIMessage(content=f"Report approved and saved to {filepath}")]
    }

def revise_node(state: ResearchState) -> dict:
    """Revise the report based on feedback."""
    feedback=state['messages'][-1].content if state['messages'] else 'Improve report'
    print("[Revise] Updating report based on feedback...")
    response=model.invoke([
        SystemMessage(content=(
            "Revise this research report based on the feedback provided.\n\n"
            f"Current Report:\n{state['draft_report']}\n\n"
            f"Feedback: {feedback}"
            )),
        HumanMessage(content='Revise the report')
    ])
    return {'draft_report': response.content}

# routing finalize n revise
def routing_after_review(state: ResearchState) -> Literal['finalize', 'revise']:
    if state.get('approved', False):
        return 'finalize'
    return 'revise'

# build graph

workflow = StateGraph(ResearchState)
workflow.add_node('research', research_node)
workflow.add_node("draft", draft_node)
workflow.add_node("review", review_node)
workflow.add_node("finalize", finalize_node)
workflow.add_node("revise", revise_node)

workflow.add_edge(START, 'research')
workflow.add_edge('research', 'draft')
workflow.add_edge('draft', 'review')

workflow.add_conditional_edges(
    'review',
    routing_after_review,
    {'finalize': 'finalize', 'revise': 'revise'}
)

workflow.add_edge('revise', 'review')
workflow.add_edge('finalize', END)

research_agent=workflow.compile()

# main
def main():
    topic = input("Research topic: ")

    result = research_agent.invoke({
        "messages": [HumanMessage(content=f"Research: {topic}")],
        "topic": topic,
        "search_res": "",
        "draft_report": "",
        "approved": False,
        "final_report": "",
    })

    if result.get("final_report"):
        print(f"\nDone! Report saved to: {result['final_report']}")
    else:
        print("\nReport was not approved.")


if __name__ == "__main__":
    main()