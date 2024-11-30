
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain

# The Chroma is On Github     https://github.com/NourTechNerd/data
persist_directory ="CHROMA"

embed_model = OllamaEmbeddings(model="llama3.1:8b",base_url="http://127.0.0.1:11434")
vector_store = Chroma(persist_directory=persist_directory,embedding_function=embed_model)
llm = Ollama(model="llama3.1:8b",base_url="http://127.0.0.1:11434")
retriever = vector_store.as_retriever(search_kwargs={"k":3})
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combined_docs_chain = create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever,combined_docs_chain)

st.title("RAG Chatbot")
question = st.text_input("poser votre question")
if st.button("obtenir la reponse"):
    if question:
        with st.spinner("Recherche de la réponse..."):
            response = retrieval_chain.invoke({"input":question})
            st.write("**réponse:**")
            st.write(response["answer"])
    else:
        st.write("veuillez poser une question")
