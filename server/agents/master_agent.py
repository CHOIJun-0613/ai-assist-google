# server/agents/master_agent.py
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

from .specialist_agents import (
    create_gmail_agent,
    create_calendar_agent,
    create_rag_agent,
    create_general_agent
)
from .state import AgentState
from ..tools.google_services import get_google_services_tools
from ..rag.retriever import get_rag_retriever

# --- 1. 에이전트 및 도구 준비 ---
# 각 전문 에이전트와 도구를 생성합니다.
gmail_tools = get_google_services_tools(['gmail'])
calendar_tools = get_google_services_tools(['calendar'])
rag_retriever = get_rag_retriever()

gmail_agent = create_gmail_agent(gmail_tools)
calendar_agent = create_calendar_agent(calendar_tools)
rag_agent = create_rag_agent(rag_retriever)
general_agent = create_general_agent()

# --- 2. LangGraph 노드 정의 ---

def general_agent_node(state: AgentState):
    """일반 대화 노드"""
    result = general_agent.invoke(state)
    return {"messages": [result]}

def gmail_agent_node(state: AgentState):
    """Gmail 관련 작업 처리 노드"""
    result = gmail_agent.invoke(state)
    return {"messages": [result]}

def calendar_agent_node(state: AgentState):
    """Google Calendar 관련 작업 처리 노드"""
    result = calendar_agent.invoke(state)
    return {"messages": [result]}

def rag_agent_node(state: AgentState):
    """RAG 기반 질의응답 처리 노드"""
    result = rag_agent.invoke(state)
    return {"messages": [result]}

# --- 3. 라우팅 로직 정의 ---

def route_message(state: AgentState):
    """사용자 메시지의 의도를 파악하여 적절한 전문가 에이전트로 라우팅"""
    last_message = state['messages'][-1]
    message_content = last_message.content.lower()

    if "메일" in message_content or "gmail" in message_content:
        return "gmail_agent"
    elif "일정" in message_content or "캘린더" in message_content or "calendar" in message_content:
        return "calendar_agent"
    # '보고서', '문서' 등 RAG가 필요한 키워드를 추가할 수 있습니다.
    elif "보고서 요약" in message_content or "알려줘" in message_content:
        # 이 부분은 더 정교한 LLM 기반 라우팅으로 개선할 수 있습니다.
        return "rag_agent"
    else:
        return "general_agent"

# --- 4. 그래프(Graph) 구성 ---

def get_agent_executor():
    """LangGraph 워크플로우를 생성하고 컴파일하여 실행기를 반환합니다."""
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("general_agent", general_agent_node)
    workflow.add_node("gmail_agent", gmail_agent_node)
    workflow.add_node("calendar_agent", calendar_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)

    # 진입점(Entry Point)에서 라우팅 로직으로 연결
    workflow.add_conditional_edges(
        "__start__",
        route_message,
        {
            "general_agent": "general_agent",
            "gmail_agent": "gmail_agent",
            "calendar_agent": "calendar_agent",
            "rag_agent": "rag_agent",
        },
    )

    # 각 전문가 에이전트 노드 실행 후에는 종료(END)
    workflow.add_edge("general_agent", END)
    workflow.add_edge("gmail_agent", END)
    workflow.add_edge("calendar_agent", END)
    workflow.add_edge("rag_agent", END)

    # 그래프 컴파일
    return workflow.compile()

