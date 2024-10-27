import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv

# API KEY 로드
load_dotenv()

st.title("나의 챗봇")

# 처음에 messages 생성
if "messages" not in st.session_state:
    # 대화 기록 저장 용도
    st.session_state["messages"] = []
    
# 사이드바 생성
with st.sidebar:
    # 대화 초기화 버튼
    clear_btn = st.button("대화 초기화")
    
    selected_prompt = st.selectbox(
        "프롬프트 선택",("기본","SNS 게시글","요약"), index=0
    )

# 이전 대화 출력
def print_message(): 
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지 추가
def add_message(role,message):
    st.session_state["messages"].append(ChatMessage(role=role,content=message))

# 체인 생성 함수
def create_chain(prompt_type):
    if prompt_type == "기본":
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system","당신은 친절한 AI 어시스턴스입니다."),
                ("human","#Question:\n{question}")
            ]
        )
    elif prompt_type == "SNS 게시글":
        prompt = load_prompt("prompts/sns.yaml")
    else:
        # 요약 프롬프트
        prompt = hub.pull("teddynote/chain-of-density-korean")
    
    # 프롬프트
    
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