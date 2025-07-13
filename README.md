AI 업무 자동화 비서 (ai-assist-google)
본 프로젝트는 Google AI(Gemini), Gmail, Google Calendar와 연동하여 반복적인 업무를 자동화하는 AI 비서 서비스입니다. LangChain/LangGraph를 활용한 Multi-Agent 아키텍처를 기반으로 사용자 요청의 의도를 파악하고, RAG를 통해 내부 문서를 참조하여 정확하고 풍부한 답변을 제공합니다. 사용자는 Streamlit으로 구현된 웹 UI를 통해 AI 비서와 대화할 수 있습니다.

🚀 프로젝트 아키텍처
Frontend (Streamlit): 사용자가 AI와 상호작용하는 웹 인터페이스입니다. 사용자의 메시지를 백엔드 API로 전송하고, 스트리밍 응답을 받아 화면에 표시합니다.

Backend (FastAPI): Streamlit 앱의 요청을 받아 처리하는 API 서버입니다.

Master Agent (LangGraph): 사용자의 질문을 가장 먼저 받아 의도를 분석하고, 어떤 전문가 에이전트(Specialist Agent)에게 작업을 위임할지 결정하는 오케스트레이터 역할을 합니다.

Specialist Agents (LangChain): 특정 도메인의 작업을 수행하는 에이전트들입니다.

Gmail Agent: Google Gmail API를 사용하여 메일 요약, 검색, 분류 등의 작업을 수행합니다.

Calendar Agent: Google Calendar API를 사용하여 일정 조회, 요약 등의 작업을 수행합니다.

RAG Agent: FAISS 기반의 Vector DB에 저장된 내부 문서(보고서, 매뉴얼 등)를 검색하여 질문에 답변합니다.

Tools: 각 에이전트가 사용하는 도구입니다. 실제 Google API를 호출하거나 Vector DB를 검색하는 함수들로 구성됩니다.

Vector Store (FAISS): RAG를 위해 텍스트 문서들이 임베딩되어 저장되는 공간입니다.

📁 디렉토리 구조
ai-assist-google/
├── app/
│   ├── app.py              # Streamlit 프론트엔드 애플리케이션
├── server/
│   ├── api/
│   │   └── chat.py         # /chat API 엔드포인트 (FastAPI 라우터)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── master_agent.py # 작업을 라우팅하는 메인 에이전트 (LangGraph)
│   │   ├── specialist_agents.py # Gmail, Calendar, RAG 에이전트 정의
│   │   └── state.py        # LangGraph 상태 정의
│   ├── core/
│   │   └── config.py       # 환경변수 로드
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingest.py       # 문서를 로드하고 Vector DB에 저장하는 스크립트
│   │   └── retriever.py    # Vector DB에서 문서를 검색하는 로직
│   ├── tools/
│   │   ├── __init__.py
│   │   └── google_services.py # Gmail, Calendar API 호출 함수 (Tools)
│   ├── main.py             # FastAPI 애플리케이션 진입점
├── vector_store/
│   └── faiss_index/        # FAISS 인덱스가 저장될 디렉토리
│       └── .gitkeep
├── documents/
│   └── sample_report.txt   # RAG로 사용할 샘플 문서
├── .env.example            # 환경변수 설정 예시 파일
└── README.md               # 프로젝트 안내 문서
└── requirements.txt        # 파이썬 패키지

셋업 및 실행 방법
1. Google Cloud 및 API 설정
Google Cloud Project 생성: Google Cloud Console에서 새 프로젝트를 생성합니다.

API 활성화: 생성한 프로젝트에서 다음 API를 활성화합니다.

Vertex AI API (또는 Generative Language API)

Gmail API

Google Calendar API

API 키 생성: API 및 서비스 > 사용자 인증 정보에서 API 키를 생성합니다. 이 키는 Google AI 모델 호출에 사용됩니다.

OAuth 2.0 클라이언트 ID 생성:

API 및 서비스 > OAuth 동의 화면에서 필요한 정보를 입력하고 동의 화면을 구성합니다. (테스트 목적일 경우 '테스트 사용자'에 본인 Google 계정 추가)

사용자 인증 정보에서 OAuth 2.0 클라이언트 ID를 생성합니다.

생성된 인증 정보(JSON 파일)를 다운로드하여 프로젝트 루트에 credentials.json 이름으로 저장합니다.

2. 프로젝트 환경 설정
저장소 복제 및 .env 파일 생성:

git clone <repository_url>
cd ai-assist-google
cp .env.example .env

.env 파일에 1단계에서 얻은 정보를 입력합니다.

3. RAG 데이터 준비
documents 디렉토리에 AI가 참조할 텍스트 파일(예: .txt, .pdf, .md)을 추가합니다.

아래 명령어를 실행하여 문서를 Vector DB에 저장합니다.

# (가상환경 활성화 후)
pip install -r server/requirements.txt
python -m server.rag.ingest
```vector_store/faiss_index` 디렉토리에 인덱스 파일이 생성됩니다.


4. 애플리케이션 실행
백엔드 서버 실행:

# (가상환경 활성화 후)
# pip install -r server/requirements.txt # 이미 설치했다면 생략
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

프론트엔드 앱 실행 (새 터미널에서):

# (가상환경 활성화 후)
pip install -r app/requirements.txt
streamlit run app/app.py

웹 브라우저에서 http://localhost:8501 주소로 접속하여 AI 비서와 대화를 시작합니다.
