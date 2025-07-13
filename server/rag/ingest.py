# server/rag/ingest.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from server.core.config import settings

def main():
    """
    'documents' 디렉토리의 문서를 로드, 분할, 임베딩하여 FAISS 벡터 저장소에 저장합니다.
    """
    print("문서 로드를 시작합니다...")
    # DirectoryLoader를 사용하여 지정된 디렉토리의 모든 .txt 파일을 로드합니다.
    loader = DirectoryLoader(
        settings.DOCUMENT_SOURCE_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    if not documents:
        print("로드할 문서가 없습니다. 'documents' 디렉토리를 확인하세요.")
        return

    print(f"총 {len(documents)}개의 문서를 로드했습니다.")

    print("텍스트 분할을 시작합니다...")
    # 텍스트를 의미 있는 단위로 분할합니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"총 {len(docs)}개의 청크(chunk)로 분할되었습니다.")

    print("임베딩 및 벡터 저장소 생성을 시작합니다...")
    # Google의 임베딩 모델을 초기화합니다.
    embeddings = GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.EMBEDDING_MODEL_NAME}",
        google_api_key=settings.GOOGLE_API_KEY
    )

    # 분할된 문서와 임베딩 모델을 사용하여 FAISS 벡터 저장소를 생성합니다.
    db = FAISS.from_documents(docs, embeddings)

    # 생성된 벡터 저장소를 지정된 경로에 저장합니다.
    if not os.path.exists(settings.VECTOR_STORE_PATH):
        os.makedirs(settings.VECTOR_STORE_PATH)
    db.save_local(settings.VECTOR_STORE_PATH)
    print(f"벡터 저장소가 '{settings.VECTOR_STORE_PATH}' 경로에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    main()

