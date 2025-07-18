from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


def build_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")
    chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True
    )
    return chain
