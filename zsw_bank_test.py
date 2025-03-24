import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


# 환경 변수 설정
OPENAI_API_KEY = "[API 키 입력]"
#DATA_PATH = "./data_kb"
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"

# def load_documents():
#     """PDF 문서를 로드하여 텍스트를 벡터 DB에 저장"""
#     documents = []
#     for file in os.listdir(DATA_PATH):
#         if file.endswith(".pdf"):
#             print(f"📄 PDF 로드 중: {file}")
#             loader = PyPDFLoader(os.path.join(DATA_PATH, file))
#             docs = loader.load()
#             print(f"🔹 {file}에서 {len(docs)}개의 문서 로드 완료")
#             documents.extend(loader.load())
#     return documents

def load_documents():
    """하위 폴더까지 포함하여 PDF 문서를 로드하고 벡터 DB에 저장"""
    documents = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                print(f"📄 PDF 로드 중: {pdf_path}")

                loader = PyPDFLoader(pdf_path)
                docs = loader.load()

                print(f"🔹 {file}에서 {len(docs)}개의 문서 로드 완료")
                documents.extend(docs)
    
    return documents

def create_chroma_db():
    """ChromaDB 인덱스를 생성하고 문서를 저장"""
    documents = load_documents()
    print(f"📚 총 {len(documents)}개의 문서가 로드되었습니다.")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"🔍 문서 분할 완료: 총 {len(docs)}개의 청크 생성됨")
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # text-embedding-ada-002
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    print("✅ ChromaDB 저장 완료!")
    return db

def get_qa_chain_with_memory(db):
    """과거 대화 내용을 기억하는 QA 체인 생성"""
    retriever = db.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.5)
    
    # 대화 기록을 유지하는 메모리 설정
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        chain_type="stuff",
        memory=memory
    )

    # 검색된 문서를 출력하는 커스텀 실행 함수
    def custom_run(query):
        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"\n🔎 질문: {query}")
        print(f"📂 검색된 문서 개수: {len(retrieved_docs)}")

        # 은행별로 대출 상품 정리
        bank_products = {}
        sources = set()

        for doc in retrieved_docs: 
            bank_name = doc.metadata.get("source", "알 수 없음").split(os.sep)[-2]
            sources.add(doc.metadata.get("source", "알 수 없음"))

            if bank_name not in bank_products:
                bank_products[bank_name] = []

            bank_products[bank_name].append(doc.page_content[:700]) # 최대  700자 까지 저장

        # 은행별 대출 상품 정보를 하나의 텍스트로 구성
        bank_info_text = "**은행별 대출 상품 정보:**\n"
        for bank, products in bank_products.items():
            bank_info_text += f"\n🏦 **{bank.upper()}**\n"
            for i, product in enumerate(products, start=1):
                bank_info_text += f"{i}. {product}...\n"

        # QA Chain 실행 (은행별 정보 포함)
        modified_query = f"{query}\n\n{bank_info_text}"
        print(modified_query)
        response = qa_chain.run(modified_query)

        # 최종 응답 정리
        response_text = f"💡 **AI 답변:**\n{response}\n\n📌 **출처:** {', '.join(sources)}"
        return response_text

        # # 출처 (참고한 PDF 파일 리스트)
        # sources = list(set(doc.metadata.get("source", "알 수 없음") for doc in retrieved_docs))
        
        # for i, doc in enumerate(retrieved_docs):
        #     print(f"📑 문서 {i+1}: {doc.page_content[:200]}... (출처: {doc.metadata.get('source', '알 수 없음')})")

        # # QA 실행
        # response = qa_chain.run(query)

        # # 답변이 문자열이라면 그대로 사용, 딕셔너리라면 "answer" 키로부터 값 추출
        # if isinstance(response, str):
        #     response_text = response
        # else:
        #     response_text = response.get("answer", "답변을 찾을 수 없습니다.")

        # response_text += f"\n\n📌 **출처:** {', '.join(sources)}"  # 출처 정보 추가
        # return response_text

    return custom_run

def main():
    st.title("💬은행 전세자금대출 Q&A 챗봇")
    st.write("은행 전세자금대출 관련 질문을 입력하세요!")

    # ChromaDB 초기화
    if not os.path.exists(CHROMA_PATH):
        st.info("문서 DB를 생성 중입니다. 잠시만 기다려 주세요...")
        db = create_chroma_db()
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    qa_chain = get_qa_chain_with_memory(db)

    user_input = st.text_input("질문을 입력하세요:")
    
    if st.button("질문하기") and user_input:
        with st.spinner("답변을 생성 중입니다..."):
            response = qa_chain(user_input)  # 과거 대화 반영됨
            st.success("📝 답변:")
            st.write(response)  # PDF 출처 포함

            # 사용자 질문 및 답변 저장
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append((user_input, response))

    # 이전 대화 기록 표시
    if "history" in st.session_state and st.session_state.history:
        st.write("### 📜 이전 대화 기록")
        for question, answer in st.session_state.history:
            st.write(f"**Q:** {question}")
            st.write(f"**A:** {answer}")
            st.write("---")

if __name__ == "__main__":
    main()
