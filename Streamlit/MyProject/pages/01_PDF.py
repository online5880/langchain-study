import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 분할기
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import glob
import os

# API KEY 로그
load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("PDF 기반 QA")

# 처음에 messages 생성
if "messages" not in st.session_state:
    # 대화 기록 저장 용도
    st.session_state["messages"] = []

# 처음에 아무것도 업로드 하지 않았을 때
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 대화 초기화 버튼
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    selected_prompt = "prompts/pdf-rag.yaml"


# 이전 대화 출력
def print_message():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role,
                                                    content=message))

# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1 : 문서 로드(Load Documents)
    # 기본적으로 페이지 단위로 분할
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Document)
    text_split = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=50)
    split_document = text_split.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    # embeddings = OpenAIEmbeddings()
    model_name = "nlpai-lab/KoE5"
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # cuda, cpu
        encode_kwargs={"normalize_embeddings": True},
    )

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 저장
    vectorstore = FAISS.from_documents(documents=split_document,
                                       embedding=hf_embeddings)

    # 단계 5 : 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성
    retriever = vectorstore.as_retriever()
    return retriever


# 체인 생성 함수
def create_chain(retriever):
    # 프롬프트 적용

    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    You must in include 'page' number in your answer
    Answer in Korean.

    #Context: 
    {context}

    #Question:
    {question}

    #Answer:""")

    # 단계 7 : 언어모델(LLM) 생성
    # 모델(LLM) 생성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 단계 8 : 체인(Chain) 생성
    chain = ({
        "context": retriever,
        "question": RunnablePassthrough()
    }
             | prompt
             | llm
             | StrOutputParser())
    return chain


# 파일 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정)
    retriever = embed_file(file=uploaded_file)
    chain = create_chain(retriever)
    st.session_state["chain"] = chain

# 버튼이 눌리면 대화 초기화
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_message()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요.")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 입력이 들어오면
if user_input:
    
    # 체인 생성
    chain = st.session_state['chain']
    
    if chain is not None:    
        # 대화 기록 출력
        st.chat_message("user").write(user_input)
        
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)를 만들어서, 여기에 토큰을 스트리밍 출력
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화 기록 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.error("파일을 업로드 해주세요.")
