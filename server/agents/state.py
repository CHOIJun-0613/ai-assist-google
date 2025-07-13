# server/agents/state.py
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    LangGraph의 상태를 나타내는 TypedDict입니다.
    이 클래스는 그래프의 각 노드를 거치면서 정보가 어떻게 흐르고 축적되는지를 정의합니다.

    Attributes:
        messages (Annotated[List[BaseMessage], operator.add]):
            대화의 전체 흐름을 나타내는 메시지 리스트입니다.
            'operator.add'는 새로운 메시지가 기존 리스트에 추가되도록 하여,
            대화 기록이 계속 유지되게 합니다.
        history (list):
            LangChain 모델에 전달될 이전 대화 기록입니다.
            주로 (human_message, ai_message) 튜플의 리스트 형태로 사용됩니다.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    history: list
