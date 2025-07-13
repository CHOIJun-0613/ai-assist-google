# server/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.api import chat

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="AI Assist Google - Backend",
    description="AI 업무 자동화 비서를 위한 백엔드 API 서버",
    version="1.0.0"
)

# CORS 미들웨어 설정
# 개발 환경에서는 모든 오리진을 허용하여 Streamlit 앱과의 통신을 원활하게 합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 프로덕션 환경에서는 보안을 위해 특정 도메인만 허용하세요.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 포함
app.include_router(chat.router, prefix="/api", tags=["Chat"])

@app.get("/", tags=["Root"])
async def read_root():
    """
    루트 엔드포인트. API 서버가 정상적으로 실행 중인지 확인합니다.
    """
    return {"message": "AI Assist Google Backend is running."}

if __name__ == "__main__":
    # 이 파일을 직접 실행할 경우 Uvicorn 서버를 시작합니다.
    # 실제 배포 시에는 'uvicorn server.main:app --host 0.0.0.0 --port 8000' 명령어를 사용합니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)

