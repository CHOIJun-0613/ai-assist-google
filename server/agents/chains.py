# server/agents/chains.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from server.core.config import settings

# --- 공통 LLM 초기화 ---
llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL_NAME,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0.7
)

def create_tool_summarizer_chain(tool, prompt_template):
    """
    주어진 도구를 먼저 실행하고, 그 결과를 LLM에 전달하여 요약하는 체인을 생성합니다.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # LCEL (LangChain Expression Language)을 사용한 체인 구성
    # 1. 'input'을 받아 tool을 실행하고, 그 결과를 'tool_output'에 저장
    # 2. 'input'과 'tool_output'을 프롬프트에 전달하여 LLM 호출
    # 3. LLM의 출력을 문자열로 파싱
    chain = (
        RunnablePassthrough.assign(
            tool_output=lambda x: tool(x["input"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def get_gmail_chain(tool):
    """Gmail 요약 체인을 생성합니다."""
    prompt_template = """
    당신은 사용자의 이메일 요약을 돕는 AI 비서입니다.
    아래는 사용자의 이메일 검색 결과입니다. 이 내용을 바탕으로 사용자에게 친절하게 요약해서 전달해주세요.
    만약 검색 결과가 없다면, "검색된 메일이 없습니다." 라고 답변해주세요.

    [이메일 검색 결과]
    {tool_output}

    [사용자 원본 질문]
    {input}

    요약 답변:
    """
    return create_tool_summarizer_chain(tool, prompt_template)

def get_calendar_chain(tool):
    """Google Calendar 요약 체인을 생성합니다."""
    prompt_template = """
    당신은 사용자의 일정을 알려주는 AI 비서입니다.
    아래는 사용자의 오늘 일정 검색 결과입니다. 이 내용을 바탕으로 사용자에게 친절하게 정리해서 전달해주세요.
    만약 일정이 없다면, "오늘 예정된 일정이 없습니다." 라고 답변해주세요.

    [오늘 일정 검색 결과]
    {tool_output}

    [사용자 원본 질문]
    {input}

    요약 답변:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # LCEL 체인을 구성합니다.
    # 1. 입력을 무시하고 tool({})을 호출하여 'tool_output'을 생성합니다.
    # 2. 원본 입력에서 'input' 키의 값을 가져옵니다.
    # 3. 위 두 값을 프롬프트에 전달하여 LLM을 호출합니다.
    # 4. LLM의 출력을 문자열로 파싱합니다.
    chain = (
        {
            "tool_output": lambda x: tool({}),
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def get_general_chain():
    """일반 대화 체인을 생성합니다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절하고 유능한 AI 비서입니다. 사용자의 질문에 명확하고 간결하게 답변하세요."),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("user", "{input}"),
    ])
    return prompt | llm | StrOutputParser()
