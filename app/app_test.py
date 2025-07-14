# app.py

import streamlit as st
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- 설정 (원래는 server.core.config 같은 파일에서 관리) ---
# 이 예제에서는 편의상 app.py에 직접 정의합니다.
# 실제 프로젝트에서는 별도 파일로 분리하여 관리하세요.
class Settings:
    # 🚨 본인의 Google Cloud Console에서 발급받은 client_secret.json 파일 경로를 지정해야 합니다.
    GOOGLE_CREDENTIALS_PATH: str = "credentials.json" 
    # 🚨 Google Cloud Console의 '승인된 리디렉션 URI'에 등록된 주소와 정확히 일치해야 합니다.
    REDIRECT_URI: str = "http://localhost:8501"
    # 사용자의 인증 토큰이 저장될 파일 경로입니다.
    TOKEN_PATH: str = "token.json"

settings = Settings()

# --- Google API 연동 설정 ---
SCOPES = {
    'gmail': ['https://www.googleapis.com/auth/gmail.readonly'],
    'calendar': ['https://www.googleapis.com/auth/calendar.readonly']
}

# --- 웹 앱용 인증 함수 (수정된 버전) ---
def get_credentials_for_webapp(service_names):
    """
    Streamlit 웹 앱 환경에서 Google OAuth 2.0 인증을 처리하고,
    유효한 Credentials 객체를 반환하거나 인증 과정을 안내합니다.
    """
    token_path = settings.TOKEN_PATH
    required_scopes = [scope for service in service_names for scope in SCOPES.get(service, [])]

    # 1. 세션에 저장된 유효한 토큰 확인
    creds = st.session_state.get('google_creds')
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        st.session_state['google_creds'] = creds
        return creds

    # 2. 파일에 저장된 토큰 확인 (앱 최초 실행 시)
    if os.path.exists(token_path) and 'google_creds' not in st.session_state:
        try:
            creds = Credentials.from_authorized_user_file(token_path, required_scopes)
            if creds and creds.valid:
                st.session_state['google_creds'] = creds
                return creds
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                st.session_state['google_creds'] = creds
                return creds
        except Exception as e:
            # 파일이 손상되었거나 스코프가 변경된 경우, 파일을 삭제하고 새로 인증
            st.error(f"토큰 파일 처리 중 오류: {e}. 재인증을 위해 토큰 파일을 삭제합니다.")
            if os.path.exists(token_path):
                os.remove(token_path)


    # 3. 인증이 필요한 경우
    try:
        flow = Flow.from_client_secrets_file(
            settings.GOOGLE_CREDENTIALS_PATH,
            scopes=required_scopes,
            redirect_uri=settings.REDIRECT_URI
        )

        query_params = st.query_params
        auth_code = query_params.get("code")

        if not auth_code:
            authorization_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent' # 사용자가 항상 동의하도록 유도
            )
            st.session_state['oauth_state'] = state
            
            st.warning("AI 에이전트의 Google 서비스 사용을 위해 계정 인증이 필요합니다.")
            st.markdown(f'👇 아래 링크를 클릭하여 Google 계정으로 로그인 해주세요.\n\n[Google 계정으로 로그인하기]({authorization_url})', unsafe_allow_html=True)
            return None
        else:
            if 'oauth_state' not in st.session_state or st.session_state['oauth_state'] != query_params.get("state"):
                st.error("인증 상태(State)가 유효하지 않습니다. 다시 시도해주세요.")
                return None

            flow.fetch_token(code=auth_code)
            creds = flow.credentials

            st.session_state['google_creds'] = creds
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
            
            # 인증 후 URL에서 code와 state를 제거하기 위해 페이지를 새로고침합니다.
            st.rerun()

    except FileNotFoundError:
        st.error(f"'{settings.GOOGLE_CREDENTIALS_PATH}' 파일을 찾을 수 없습니다. Google Cloud Console에서 다운로드하여 프로젝트 루트에 배치해주세요.")
        return None
    except Exception as e:
        st.error(f"인증 과정에서 오류가 발생했습니다: {e}")
        return None

# --- LangChain Tool 정의 ---
# @tool 데코레이터를 사용하려면 langchain 라이브러리가 설치되어 있어야 합니다.
# pip install langchain
from langchain.tools import tool

@tool
def search_gmail(query: str) -> str:
    """주어진 쿼리로 Gmail을 검색하여 최근 5개 메일의 제목과 보낸 사람 목록을 반환합니다."""
    # st.session_state에서 인증 정보를 가져옵니다.
    creds = st.session_state.get('google_creds')
    if not creds:
        return "Gmail을 검색하려면 먼저 Google 계정 인증이 필요합니다."
    
    try:
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', q=query, maxResults=5).execute()
        messages = results.get('messages', [])

        if not messages:
            return "해당 쿼리에 대한 메일을 찾을 수 없습니다."

        email_list = []
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = msg_data['payload']['headers']
            subject = next((header['value'] for header in headers if header['name'] == 'Subject'), '제목 없음')
            sender = next((header['value'] for header in headers if header['name'] == 'From'), '발신자 불명')
            email_list.append(f"**제목:** {subject}\n**보낸 사람:** {sender}\n---")
        
        return "\n".join(email_list)
    except HttpError as error:
        return f"Gmail API 호출 중 오류 발생: {error}"
    except Exception as e:
        return f"알 수 없는 오류 발생: {e}"

# --- Streamlit 메인 애플리케이션 ---
def main():
    st.set_page_config(page_title="AI 에이전트", layout="centered")
    st.title("🤖 AI 에이전트")

    # 1. 앱 시작 시 인증 처리
    # 사용할 구글 서비스를 리스트로 전달합니다.
    creds = get_credentials_for_webapp(['gmail', 'calendar'])

    # 2. 인증이 완료되었을 때만 에이전트 기능 활성화
    if creds:
        st.success(f"**{creds.id_token.get('name', '사용자')}님**, 안녕하세요! Google 계정 인증이 완료되었습니다.")
        
        st.info("이제 Gmail 검색 등 연동된 Google 서비스를 사용할 수 있습니다.")

        # Gmail 검색 UI
        with st.form("gmail_search_form"):
            query = st.text_input("Gmail에서 검색할 내용을 입력하세요:", placeholder="예: AI 관련 최신 뉴스")
            submitted = st.form_submit_button("검색")

            if submitted and query:
                with st.spinner("Gmail을 검색하는 중..."):
                    search_result = search_gmail(query=query)
                    st.markdown("### 📧 Gmail 검색 결과")
                    st.markdown(search_result)

    else:
        # 인증이 완료되지 않았으면, get_credentials_for_webapp 함수가
        # 로그인 링크를 화면에 표시해주므로 여기서는 대기 메시지만 보여줍니다.
        st.info("Google 서비스 연동을 위해 화면의 안내에 따라 로그인을 진행해주세요.")

if __name__ == "__main__":
    main()
