# AI 업무 자동화 비서 (ai-assist-google)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-purple.svg)](https://www.langchain.com/)

**AI 업무 자동화 비서**는 Google AI (Gemini), Gmail, Google Calendar와 연동하여 반복적인 업무를 자동화하는 지능형 AI 비서 서비스입니다. LangChain과 LangGraph를 활용한 Multi-Agent 아키텍처를 기반으로, 사용자의 복잡한 요청을 이해하고 최적의 도구를 사용하여 작업을 수행합니다. 또한, RAG(Retrieval-Augmented Generation) 기술을 통해 내부 문서를 참조하여 정확하고 풍부한 답변을 제공합니다.

## 🌟 주요 기능

- **지능형 작업 라우팅**: 사용자의 질문 의도를 분석하여 Gmail, Google Calendar, 내부 문서 검색 등 가장 적합한 에이전트에게 작업을 동적으로 위임합니다.
- **Google 서비스 연동**:
    - **Gmail**: 이메일 요약, 검색, 분류 및 답장 초안 작성
    - **Google Calendar**: 일정 조회, 생성 및 요약
- **내부 문서 기반 답변 (RAG)**: `documents` 폴더 내의 회사 보고서, 매뉴얼, 회의록 등을 기반으로 질문에 답변하여 정보의 정확성을 높입니다.
- **대화형 웹 인터페이스**: Streamlit으로 구현된 직관적인 UI를 통해 사용자와 AI 비서가 실시간으로 대화할 수 있습니다.

## 🏗️ 시스템 아키텍처

본 프로젝트는 다음과 같은 모듈식 아키텍처로 구성되어 있습니다.

1.  **Frontend (Streamlit)**: 사용자가 AI와 상호작용하는 웹 인터페이스입니다. 사용자의 메시지를 백엔드 API로 전송하고, 스트리밍 응답을 받아 화면에 표시합니다.
2.  **Backend (FastAPI)**: Streamlit 앱의 요청을 받아 처리하는 API 서버입니다.
3.  **Master Agent (LangGraph)**: 사용자의 질문을 가장 먼저 받아 의도를 분석하고, 어떤 전문가 에이전트(Specialist Agent)에게 작업을 위임할지 결정하는 오케스트레이터 역할을 합니다.
4.  **Specialist Agents (LangChain)**: 특정 도메인의 작업을 수행하는 에이전트들입니다.
    - **Gmail Agent**: Google Gmail API를 사용하여 메일 관련 작업을 수행합니다.
    - **Calendar Agent**: Google Calendar API를 사용하여 일정 관련 작업을 수행합니다.
    - **RAG Agent**: FAISS 기반의 Vector DB에 저장된 내부 문서를 검색하여 질문에 답변합니다.
5.  **Tools**: 각 에이전트가 사용하는 도구입니다. 실제 Google API를 호출하거나 Vector DB를 검색하는 함수들로 구성됩니다.
6.  **Vector Store (FAISS)**: RAG를 위해 텍스트 문서들이 임베딩되어 저장되는 공간입니다.

## 🚀 시작하기

### 1. 사전 준비: Google Cloud 및 API 설정

1.  **Google Cloud 프로젝트 생성**: [Google Cloud Console](https://console.cloud.google.com/)에서 새 프로젝트를 생성합니다.
2.  **API 활성화**: 생성한 프로젝트에서 다음 API를 활성화합니다.
    - Vertex AI API (또는 Generative Language API)
    - Gmail API
    - Google Calendar API
3.  **API 키 생성**: `API 및 서비스 > 사용자 인증 정보`에서 **API 키**를 생성합니다. 이 키는 Google AI 모델 호출에 사용됩니다.
4.  **OAuth 2.0 클라이언트 ID 생성**:
    - `API 및 서비스 > OAuth 동의 화면`에서 필요한 정보를 입력하고 동의 화면을 구성합니다. (테스트 목적일 경우 '테스트 사용자'에 본인 Google 계정 추가)
    - `사용자 인증 정보`에서 **OAuth 2.0 클라이언트 ID**를 생성합니다.
    - 생성된 인증 정보(JSON 파일)를 다운로드하여 프로젝트 루트에 `credentials.json` 이름으로 저장합니다.

### 2. 프로젝트 환경 설정

```bash
# 1. 저장소 복제
git clone https://github.com/your-username/ai-assist-google.git
cd ai-assist-google

# 2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. .env 파일 생성 및 설정
cp .env.example .env
```

-   생성된 `.env` 파일을 열고, 1단계에서 얻은 **Google API 키**와 **프로젝트 ID**를 입력합니다.

### 3. RAG 데이터 준비 및 인덱싱

1.  `documents` 디렉토리에 AI가 참조할 텍스트 파일(예: `.txt`, `.pdf`, `.md`)을 추가합니다.
2.  아래 명령어를 실행하여 문서를 Vector DB에 저장합니다.

```bash
# 필요한 라이브러리 설치
pip install -r requirements.txt

# 데이터 인덱싱 실행
python -m server.rag.ingest
```

-   `vector_store/faiss_index` 디렉토리에 인덱스 파일(`index.faiss`, `index.pkl`)이 생성됩니다.

### 4. 애플리케이션 실행

1.  **백엔드 서버 실행**:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

2.  **프론트엔드 앱 실행** (새 터미널에서):

```bash
streamlit run app/app.py
```

3.  웹 브라우저에서 `http://localhost:8501` 주소로 접속하여 AI 비서와 대화를 시작합니다.

## 📁 디렉토리 구조

```
ai-assist-google/
├── app/
│   └── app.py              # Streamlit 프론트엔드
├── server/
│   ├── api/
│   │   └── chat.py         # FastAPI /chat 엔드포인트
│   ├── agents/
│   │   ├── master_agent.py # 작업 라우팅 에이전트 (LangGraph)
│   │   └── specialist_agents.py # 전문가 에이전트 (Gmail, Calendar, RAG)
│   ├── core/
│   │   └── config.py       # 환경변수 관리
│   ├── rag/
│   │   ├── ingest.py       # 문서 인덱싱 스크립트
│   │   └── retriever.py    # 문서 검색 로직
│   ├── tools/
│   │   └── google_services.py # Google API 호출 도구
│   └── main.py             # FastAPI 앱 진입점
├── vector_store/
│   └── faiss_index/        # FAISS 인덱스 저장소
├── documents/
│   └── *.txt               # RAG용 소스 문서
├── .env.example            # 환경변수 예시
├── credentials.json        # Google OAuth 2.0 인증 정보
├── requirements.txt        # Python 패키지 목록
└── README.md               # 프로젝트 안내 문서
```

## 🛠️ 기술 스택

-   **언어**: Python 3.9+
-   **AI/ML**: LangChain, LangGraph, Google Gemini, FAISS, TikToken
-   **백엔드**: FastAPI, Uvicorn
-   **프론트엔드**: Streamlit
-   **Google API**: Google AI, Gmail, Google Calendar
-   **기타**: python-dotenv, Pydantic