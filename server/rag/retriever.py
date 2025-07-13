# server/rag/retriever.py
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from server.core.config import settings

def get_rag_retriever():
    """
    미리 생성된 FAISS 인덱스를 로컬 경로에서 불러와,
    LangChain에서 사용할 수 있는 Retriever 객체를 생성하고 반환합니다.

    Retriever는 사용자 쿼리가 주어졌을 때, 벡터 저장소에서 가장 유사한
    문서 청크(chunk)들을 검색하는 역할을 담당합니다.

    Returns:
        langchain.schema.retriever.BaseRetriever: LangChain 호환 Retriever 객체.

    Raises:
        FileNotFoundError: 지정된 경로에 벡터 저장소 파일이 없을 경우 발생합니다.
                           이 경우, 'python -m server.rag.ingest'를 먼저 실행해야 합니다.
    """
    # 벡터 저장소 경로 존재 여부 확인
    if not os.path.exists(settings.VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {settings.VECTOR_STORE_PATH}. "
            "Please run 'python -m server.rag.ingest' first to create it."
        )

    # 임베딩 모델 초기화 (ingest 시 사용했던 모델과 동일해야 함)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.EMBEDDING_MODEL_NAME}",
        google_api_key=settings.GOOGLE_API_KEY
    )
    
    # 로컬에 저장된 FAISS 인덱스를 메모리로 로드
    # allow_dangerous_deserialization=True는 pickle 기반으로 저장된
    # FAISS 인덱스를 로드할 때 필요한 옵션입니다. 신뢰할 수 있는
    # 인덱스 파일에만 사용해야 합니다.
    db = FAISS.load_local(
        settings.VECTOR_STORE_PATH, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # 로드된 벡터 저장소를 LangChain의 Retriever로 변환하여 반환
    # as_retriever()는 기본적으로 유사도 검색을 수행합니다.
    return db.as_retriever()
