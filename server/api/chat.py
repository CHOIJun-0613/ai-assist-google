# server/api/chat.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Tuple
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
        
        # 사용자의 메시지와 대화 기록을 입력으로 에이전트를 실행합니다.
        # LangChain의 invoke 메소드는 에이전트의 최종 결과를 반환합니다.
        result = agent_executor.invoke({
            "messages": [("user", request.message)],
            "history": request.history
        })
        
        # 결과에서 AI의 마지막 응답을 추출합니다.
        # LangGraph의 결과는 'messages' 키에 대화 흐름 전체를 담고 있습니다.
        ai_response = result['messages'][-1].content
        
        return ChatResponse(response=ai_response)
    except Exception as e:
        # 에러 발생 시 로그를 남기고, 사용자에게 에러 메시지를 반환할 수 있습니다.
        print(f"Error during chat processing: {e}")
        return ChatResponse(response=f"죄송합니다, 요청을 처리하는 중 오류가 발생했습니다: {e}")

