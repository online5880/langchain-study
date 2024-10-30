import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import glob
import os

# API KEY 로그
load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir('.cache')

# 파일 업로드 전용 폴더    
if not os.path.exists(".cache/files"):
    os.mkdir('.cache/files')
     
if not os.path.exists(".cache/embeddings"):
    os.mkdir('.cache/embeddings')   

st.title("PDF 기반 QA")

# 처음에 messages 생성
if "messages" not in st.session_state:
    # 대화 기록 저장 용도
    st.session_state["messages"] = []
    
# 사이드바 생성
with st.sidebar:
    # 대화 초기화 버튼
    clear_btn = st.button("대화 초기화")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드",type=["pdf"])
    
    selected_prompt = "prompts/pdf-rag.yaml"
    

# 이전 대화 출력
def print_message(): 
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지 추가
def add_message(role,message):
    st.session_state["messages"].append(ChatMessage(role=role,content=message))
    
# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, 'wb') as f:
        f.write(file_content)
        
# 파일 업로드 되었을 때
if uploaded_file:
    embed_file(file=uploaded_file)
    
# 체인 생성 함수
def create_chain(prompt_filepath):
    # 프롬프트 적용
    prompt = load_prompt(prompt_filepath)
    
    # GPT
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    # ollama
    # llm = ChatOllama(model="Lama3.2-korean:latest", temperature=0)
    
    # 출력 파서
    output_parser = StrOutputParser()
    
    # 체인 생성
    chain = prompt | llm | output_parser
    return chain

# 버튼이 눌리면 대화 초기화
if clear_btn:
    st.session_state["messages"] = []
    
# 이전 대화 기록 출력
print_message()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요.")

# 입력이 들어오면
if user_input:
    # 대화 기록 출력
    st.chat_message("user").write(user_input)
    # 체인 생성
    chain = create_chain(selected_prompt)
    # 스트리밍 호출
    response = chain.stream({"question":user_input})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)를 만들어서, 여기에 토큰을 스트리밍 출력
        container = st.empty()
        
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
        
    
    # 대화 기록 저장  
    add_message("user",user_input)
    add_message("assistant",ai_answer)