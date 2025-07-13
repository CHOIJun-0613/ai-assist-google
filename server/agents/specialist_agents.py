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
    temperature=0.7
)

# --- 1. 일반 대화 에이전트 ---
def create_general_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절하고 유능한 AI 비서입니다. 사용자의 질문에 명확하고 간결하게 답변하세요."),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("user", "{input}"),
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
        MessagesPlaceholder(variable_name="history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor

# --- 3. 전문가 에이전트 생성 (프롬프트 수정됨) ---
def create_gmail_agent(tools):
    # --- FIX: 도구 사용을 강력하게 지시하는 프롬프트로 변경 ---
    system_prompt = """
    당신은 Gmail API와 직접 연결된 AI 에이전트입니다.
    당신의 유일한 임무는 사용자의 질문에 답하기 위해 'search_gmail' 도구를 사용하는 것입니다.
    절대 추측하거나 일반적인 지식으로 답변해서는 안 됩니다. "메일을 확인할 수 없다" 또는 "권한이 없다"와 같은 답변을 해서는 안됩니다.
    사용자가 메일에 대해 질문하면, 당신은 반드시 'search_gmail' 도구를 호출하여 그 결과를 바탕으로 답변해야 합니다. 이것이 당신의 유일한 기능입니다.
    """
    return create_react_agent(tools, system_prompt)

def create_calendar_agent(tools):
    # --- FIX: 도구 사용을 강력하게 지시하는 프롬프트로 변경 ---
    system_prompt = """
    당신은 Google Calendar API와 직접 연결된 AI 에이전트입니다.
    당신의 유일한 임무는 사용자의 질문에 답하기 위해 'get_today_calendar_events' 도구를 사용하는 것입니다.
    절대 추측하거나 일반적인 지식으로 답변해서는 안 됩니다. "일정을 확인할 수 없다" 또는 "권한이 없다"와 같은 답변을 해서는 안됩니다.
    사용자가 일정에 대해 질문하면, 당신은 반드시 'get_today_calendar_events' 도구를 호출하여 그 결과를 바탕으로 답변해야 합니다. 이것이 당신의 유일한 기능입니다.
    """
    return create_react_agent(tools, system_prompt)

# --- 4. RAG 에이전트 (Retriever-Augmented Generation) ---
def create_rag_agent(retriever):
    system_prompt = (
        "당신은 문서 검색 및 요약 전문가입니다."
        "주어진 문서(context)를 기반으로 사용자의 질문에 답변하세요."
        "문서에 없는 내용은 답변하지 말고, 정보가 없다고 솔직하게 말하세요."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("user", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    def invoke_rag_chain(state):
        result = rag_chain.invoke({
            "input": state["input"], 
            "history": state.get("history", [])
        })
        return result.get('answer', "죄송합니다, 답변을 생성할 수 없습니다.")

    return invoke_rag_chain