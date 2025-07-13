# server/tools/google_services.py
import os
from langchain.tools import tool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from server.core.config import settings
import base64
import email

# --- Google API 연동 설정 ---
# 필요한 권한 범위(SCOPES)를 정의합니다.
SCOPES = {
    'gmail': ['https://www.googleapis.com/auth/gmail.readonly'],
    'calendar': ['https://www.googleapis.com/auth/calendar.readonly']
}

def get_credentials(service_names):
    """
    사용자 인증을 처리하고 유효한 Credentials 객체를 반환합니다.
    토큰이 없거나 만료된 경우, OAuth 2.0 흐름을 통해 새로 발급받습니다.
    """
    creds = None
    token_path = settings.TOKEN_PATH
    
    required_scopes = []
    for service in service_names:
        required_scopes.extend(SCOPES.get(service, []))
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, required_scopes)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # 이 부분은 Streamlit/FastAPI 환경에서 개선이 필요합니다.
            # 초기 실행 시 터미널에서 인증을 수행해야 합니다.
            flow = InstalledAppFlow.from_client_secrets_file(
                settings.GOOGLE_CREDENTIALS_PATH, required_scopes,
                # redirect_uri=settings.REDIRECT_URI # 웹 앱에서는 리디렉션 URI 설정 필요
            )
            creds = flow.run_local_server(port=0)
        
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return creds

# --- LangChain Tool 정의 ---

@tool
def search_gmail(query: str) -> str:
    """
    "주어진 쿼리로 Gmail을 검색하여 최근 5개 메일의 제목과 보낸 사람 목록을 반환합니다.
    예: 'AI 관련 최신 뉴스'
    """
    try:
        creds = get_credentials(['gmail'])
        service = build('gmail', 'v1', credentials=creds)
        
        results = service.users().messages().list(userId='me', q=query, maxResults=5).execute()
        messages = results.get('messages', [])

        if not messages:
            return "해당 쿼리에 대한 메일을 찾을 수 없습니다."

        email_list = []
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = msg_data['payload']['headers']
            subject = next(header['value'] for header in headers if header['name'] == 'Subject')
            sender = next(header['value'] for header in headers if header['name'] == 'From')
            email_list.append(f"제목: {subject}\n보낸 사람: {sender}\n---")
        
        return "\n".join(email_list)
    except HttpError as error:
        return f"Gmail API 호출 중 오류 발생: {error}"
    except Exception as e:
        return f"알 수 없는 오류 발생: {e}"

@tool
def get_today_calendar_events() -> str:
    """
    "오늘의 Google Calendar 일정을 모두 가져와 요약해서 반환합니다."
    """
    # 이 함수는 인자가 필요 없습니다.
    # 실제 구현에서는 날짜를 인자로 받을 수 있도록 확장할 수 있습니다.
    from datetime import datetime, time, timezone, timedelta

    try:
        creds = get_credentials(['calendar'])
        service = build('calendar', 'v3', credentials=creds)

        # 'Z'는 UTC를 의미합니다. 한국 시간(KST)에 맞게 조정합니다.
        KST = timezone(timedelta(hours=9))
        now = datetime.now(KST)
        time_min = datetime.combine(now.date(), time.min, tzinfo=KST).isoformat()
        time_max = datetime.combine(now.date(), time.max, tzinfo=KST).isoformat()

        events_result = service.events().list(
            calendarId='primary', timeMin=time_min, timeMax=time_max,
            singleEvents=True, orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])

        if not events:
            return "오늘 예정된 일정이 없습니다."

        event_list = ["오늘의 일정입니다:"]
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            # 시간 포맷팅
            start_dt = datetime.fromisoformat(start)
            start_str = start_dt.strftime('%H:%M') if 'dateTime' in event['start'] else '하루 종일'
            event_list.append(f"- {start_str}: {event['summary']}")
        
        return "\n".join(event_list)
    except HttpError as error:
        return f"Calendar API 호출 중 오류 발생: {error}"
    except Exception as e:
        return f"알 수 없는 오류 발생: {e}"


def get_google_services_tools(services: list):
    """요청된 서비스에 따라 관련된 도구 리스트를 반환합니다."""
    tools = []
    if 'gmail' in services:
        tools.append(search_gmail)
    if 'calendar' in services:
        tools.append(get_today_calendar_events)
    return tools
