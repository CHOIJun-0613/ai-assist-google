# server/agents/specialist_agents.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from server.core.config import settings

# --- 공통 LLM 초기화 ---
llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL_NAME,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0.7,
    convert_system_message_to_human=True # Gemini는 system message를 user message로 변환해야 함
)

# --- 1. 일반 대화 에이전트 ---
def create_general_agent():
    """
    특별한 도구 없이 대화하는 기본 에이전트.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절하고 유능한 AI 비서입니다. 사용자의 질문에 명확하고 간결하게 답변하세요."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{messages}"),
    ])
    chain = prompt | llm
    return chain

# --- 2. ReAct (Tool-Calling) 기반 에이전트 생성 함수 ---
def create_react_agent(tools, system_prompt):
    """
    주어진 도구와 프롬프트를 사용하여 ReAct(Tool-Calling) 에이전트를 생성합니다.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{messages}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor

# --- 3. 전문가 에이전트 생성 ---
def create_gmail_agent(tools):
    """Gmail 관련 작업을 처리하는 에이전트"""
    system_prompt = "당신은 Gmail 전문가입니다. 사용자의 요청에 따라 메일을 검색하고, 요약하고, 관련 정보를 제공하세요."
    return create_react_agent(tools, system_prompt)

def create_calendar_agent(tools):
    """Google Calendar 관련 작업을 처리하는 에이전트"""
    system_prompt = "당신은 Google Calendar 전문가입니다. 사용자의 요청에 따라 일정을 조회하고, 요약하여 알려주세요."
    return create_react_agent(tools, system_prompt)

# --- 4. RAG 에이전트 (Retriever-Augmented Generation) ---
def create_rag_agent(retriever):
    """
    내부 문서를 검색하여 답변을 생성하는 RAG 에이전트.
    이 에이전트는 ReAct가 아닌 Retrieval Chain을 사용합니다.
    """
    system_prompt = (
        "당신은 문서 검색 및 요약 전문가입니다."
        "주어진 문서(context)를 기반으로 사용자의 질문에 답변하세요."
        "문서에 없는 내용은 답변하지 말고, 정보가 없다고 솔직하게 말하세요."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"), # Retrieval Chain은 'input'을 키로 사용
    ])
    
    # 문서를 프롬프트에 채워넣는 체인
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # 리트리버와 답변 생성 체인을 결합
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # LangGraph와 호환되도록 입력/출력 형식을 맞춤
    # 'messages'에서 사용자 입력을 추출하여 'input'으로 전달
    def invoke_rag_chain(state):
        last_user_message = state['messages'][-1].content
        history = state.get('history', [])
        return rag_chain.invoke({"input": last_user_message, "history": history})['answer']

    return invoke_rag_chain
