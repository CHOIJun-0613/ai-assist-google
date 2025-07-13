# server/api/chat.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Tuple
from langchain_core.messages import HumanMessage, AIMessage
from server.agents.master_agent import get_agent_executor

# API 라우터 생성
router = APIRouter()

# 요청 본문(Request Body) 모델 정의
class ChatRequest(BaseModel):
    message: str
    history: List[Tuple[str, str]] # (human_message, ai_message) 형태의 리스트

# 응답 본문(Response Body) 모델 정의
class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    사용자의 채팅 메시지를 받아 AI 에이전트의 응답을 반환하는 엔드포인트.
    """
    try:
        # LangGraph로 구성된 에이전트 실행기(executor)를 가져옵니다.
        agent_executor = get_agent_executor()
        
        # Streamlit에서 받은 튜플 형태의 history를 HumanMessage와 AIMessage로 변환합니다.
        chat_history = []
        for human, ai in request.history:
            chat_history.append(HumanMessage(content=human))
            chat_history.append(AIMessage(content=ai))
        
        # 현재 사용자 메시지를 HumanMessage로 추가합니다.
        current_message = HumanMessage(content=request.message)
        
        # AgentState는 'messages'와 'history'를 모두 가질 수 있습니다.
        # 에이전트 프롬프트가 'history'를 요구하므로, 여기서 명시적으로 전달해줍니다.
        initial_state = {
            "messages": chat_history + [current_message],
            "history": chat_history
        }

        # 에이전트를 완전한 초기 상태와 함께 실행합니다.
        result = agent_executor.invoke(initial_state)
        
        # --- FIX: 다양한 출력 형태에 대응하도록 응답 추출 로직 수정 ---
        # LangGraph의 최종 상태(state)에서 마지막 메시지를 가져옵니다.
        last_message = result['messages'][-1]
        
        ai_response = ""
        if hasattr(last_message, 'content'):
            # 마지막 메시지가 AIMessage와 같은 표준 메시지 객체인 경우
            ai_response = last_message.content
        elif isinstance(last_message, dict):
            # 마지막 메시지가 AgentExecutor의 출력값인 딕셔너리인 경우
            # 'output' 키 또는 다른 잠재적 키에서 응답을 추출합니다.
            ai_response = last_message.get('output') or last_message.get('answer') or str(last_message)
        else:
            # 그 외의 경우, 문자열로 변환합니다.
            ai_response = str(last_message)
        
        return ChatResponse(response=ai_response)
    except Exception as e:
        # 에러 발생 시 로그를 남기고, 사용자에게 에러 메시지를 반환할 수 있습니다.
        print(f"Error during chat processing: {e}")
        # 디버깅을 위해 에러 스택 트레이스를 출력합니다.
        import traceback
        traceback.print_exc()
        return ChatResponse(response=f"죄송합니다, 요청을 처리하는 중 오류가 발생했습니다: {e}")
