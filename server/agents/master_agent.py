# server/agents/master_agent.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from .state import AgentState
from ..tools.google_services import get_google_services_tools
from .chains import get_gmail_chain, get_calendar_chain, get_general_chain
# RAG 기능은 아직 사용하지 않으므로 주석 처리 (필요시 활성화)
# from ..rag.retriever import get_rag_retriever

# --- 1. 도구 및 체인 준비 ---
tools = get_google_services_tools(['gmail', 'calendar'])
gmail_tool = next((t for t in tools if t.name == 'search_gmail'), None)
calendar_tool = next((t for t in tools if t.name == 'get_today_calendar_events'), None)

# 각 작업에 맞는 체인(Chain)을 생성합니다.
gmail_chain = get_gmail_chain(gmail_tool)
calendar_chain = get_calendar_chain(calendar_tool)
general_chain = get_general_chain()
# rag_chain = ... # 필요시 RAG 체인도 여기에 정의

# --- 2. LangGraph 노드 정의 ---
def chain_node(state: AgentState, chain):
    """
    주어진 체인을 실행하고, 그 결과를 AIMessage로 변환하여 상태를 업데이트합니다.
    """
    user_input = state['messages'][-1].content
    history = state['messages'][:-1]
    
    # 체인 실행
    result = chain.invoke({
        "input": user_input,
        "history": history
    })
    
    return {"messages": [AIMessage(content=result)]}

def general_node(state: AgentState):
    return chain_node(state, general_chain)

def gmail_node(state: AgentState):
    return chain_node(state, gmail_chain)

def calendar_node(state: AgentState):
    return chain_node(state, calendar_chain)

# --- 3. 라우팅 로직 정의 ---
def route_message(state: AgentState):
    """사용자 메시지의 의도를 파악하여 적절한 체인으로 라우팅합니다."""
    last_message = state['messages'][-1]
    message_content = last_message.content.lower()

    if "메일" in message_content or "gmail" in message_content:
        print("Routing to: gmail_node")
        return "gmail_node"
    elif "일정" in message_content or "캘린더" in message_content or "calendar" in message_content:
        print("Routing to: calendar_node")
        return "calendar_node"
    else:
        print("Routing to: general_node")
        return "general_node"

# --- 4. 그래프(Graph) 구성 ---
def get_agent_executor():
    """LangGraph 워크플로우를 생성하고 컴파일하여 실행기를 반환합니다."""
    workflow = StateGraph(AgentState)

    workflow.add_node("general_node", general_node)
    workflow.add_node("gmail_node", gmail_node)
    workflow.add_node("calendar_node", calendar_node)

    workflow.set_conditional_entry_point(
        route_message,
        {
            "general_node": "general_node",
            "gmail_node": "gmail_node",
            "calendar_node": "calendar_node",
        },
    )

    workflow.add_edge("general_node", END)
    workflow.add_edge("gmail_node", END)
    workflow.add_edge("calendar_node", END)

    return workflow.compile()
