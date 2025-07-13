# app/app.py
import streamlit as st
import requests
import json

# --- 페이지 설정 ---
st.set_page_config(
    page_title="AI 업무 자동화 비서",
    page_icon="🤖",
    layout="wide"
)

# --- 백엔드 API 주소 ---
BACKEND_API_URL = "http://localhost:8000/api/chat"

# --- UI 구성 ---
st.title("🤖 AI 업무 자동화 비서")
st.markdown("""
안녕하세요! 저는 당신의 업무를 돕는 AI 비서입니다.  
메일 정리, 일정 확인, 문서 요약 등 다양한 작업을 요청해보세요.

**예시 질문:**
- "오늘 내 캘린더에 무슨 일정이 있어?"
- "AI 관련해서 새로 온 메일 찾아줘"
- "최신 AI 기술 보고서 요약해줘"
""")

# --- 세션 상태 초기화 ---
# 'messages'는 채팅 기록을 저장합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 채팅 기록 표시 ---
# 이전 대화 내용을 화면에 표시합니다.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 사용자 입력 처리 ---
# st.chat_input을 사용하여 사용자로부터 입력을 받습니다.
if prompt := st.chat_input("무엇을 도와드릴까요?"):
    # 사용자가 입력한 메시지를 채팅 기록에 추가하고 화면에 표시합니다.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI의 응답을 표시할 준비를 합니다.
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("생각 중...")

        try:
            # 백엔드 API에 요청을 보냅니다.
            # 대화 기록을 함께 보내 문맥을 유지하도록 합니다.
            history = [
                (msg["content"], st.session_state.messages[i+1]["content"])
                for i, msg in enumerate(st.session_state.messages)
                if msg["role"] == "user" and i + 1 < len(st.session_state.messages)
            ]
            
            response = requests.post(
                BACKEND_API_URL,
                json={"message": prompt, "history": history}
            )
            response.raise_for_status()  # HTTP 오류 발생 시 예외 처리

            # API 응답을 받아 화면에 표시합니다.
            ai_response = response.json()["response"]
            message_placeholder.markdown(ai_response)
            
            # AI의 응답을 채팅 기록에 추가합니다.
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

        except requests.exceptions.RequestException as e:
            message_placeholder.error(f"백엔드 서버와 통신 중 오류가 발생했습니다: {e}")
        except Exception as e:
            message_placeholder.error(f"예상치 못한 오류가 발생했습니다: {e}")

