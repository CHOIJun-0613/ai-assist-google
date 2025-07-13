# server/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    .env 파일로부터 환경 변수를 로드하는 설정 클래스입니다.
    Pydantic을 사용하여 타입 검증을 수행하고, 기본값을 설정하며,
    코드 내에서 일관된 방식으로 설정 값에 접근할 수 있도록 합니다.

    Attributes:
        GOOGLE_API_KEY (str): Google AI (Gemini) 및 기타 Google 서비스용 API 키.
        GEMINI_MODEL_NAME (str): 사용할 Gemini 모델의 이름.
        EMBEDDING_MODEL_NAME (str): RAG에 사용할 텍스트 임베딩 모델의 이름.
        VECTOR_STORE_PATH (str): 생성된 FAISS 벡터 DB가 저장될 로컬 경로.
        DOCUMENT_SOURCE_DIR (str): RAG가 참조할 원본 문서들이 위치한 디렉토리.
        GOOGLE_CREDENTIALS_PATH (str): Google OAuth 2.0 인증 정보(JSON) 파일 경로.
        TOKEN_PATH (str): 생성된 OAuth 토큰이 저장될 파일 경로.
        REDIRECT_URI (str): OAuth 2.0 인증 시 사용될 리디렉션 URI.
    """
    GOOGLE_API_KEY: str
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash-latest"
    EMBEDDING_MODEL_NAME: str = "text-embedding-004"
    VECTOR_STORE_PATH: str = "./vector_store/faiss_index"
    DOCUMENT_SOURCE_DIR: str = "./documents"
    GOOGLE_CREDENTIALS_PATH: str = "./credentials.json"
    TOKEN_PATH: str = "./token.json"
    REDIRECT_URI: str = "http://localhost:8501"

    # .env 파일을 읽도록 설정
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

# 설정 클래스의 인스턴스를 생성하여 다른 모듈에서 가져와 사용할 수 있도록 합니다.
settings = Settings()
