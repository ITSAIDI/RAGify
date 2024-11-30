from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain_core.documents import Document
import streamlit as st
import chromadb
from langchain.prompts import ChatPromptTemplate


chromadb.api.client.SharedSystemClient.clear_system_cache()

URL = "http://localhost:11434"
persist_directory="CHROMA_2"


embed_model = OllamaEmbeddings(
    model="hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F32",
    base_url=URL,
    show_progress =True
)

def Extract_pdf_content(pdf_file):
    """
    Extracts the content of each page in a PDF file and returns a list of pages.
    """
    reader = PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text())
    return pages

def Proccess_Files(Files):
    if Files : 
        st.title("📄 Reading Files ...")
        progress_percentage = 0
        Documents = []
        
        # Initialize the progress bar
        total_files = len(Files)
        progress_bar = st.progress(0)
        
        for file_index, file in enumerate(Files):
            # Extract content from the PDF
            Pages_Contents = Extract_pdf_content(file)
            file_name = file.name
            
            # Process each page
            for index, Page in enumerate(Pages_Contents):
                document = Document(
                    page_content=Page,
                    metadata={"source": file_name, "PageNum": index + 1}
                )
                Documents.append(document)
            
            # Update the progress bar
            progress_percentage = int(((file_index + 1) / total_files) * 100)
            progress_bar.progress(progress_percentage,text =f"{progress_percentage} %")

        if progress_percentage == 100:
            st.success("✅ Files processing completed!")
            st.session_state['Documents'] = Documents
        return Documents

    return None

def Chunking(documents):
    if documents :
        st.title("✂️ Chunking documents ...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=600)
        Chunks = text_splitter.split_documents(documents)
        st.write("#### Number of Chunks is :",len(Chunks))
        if Chunks :
            st.success("✅ Chunking completed!")
            st.session_state['Chunks'] = Chunks
        return Chunks
    return None

def Create_Database(Chunks):
    if Chunks :
        st.title("🗄️ Creating ChromaDB ...")
        vector_store = Chroma.from_documents(Chunks,embed_model,persist_directory=persist_directory)
        
        st.success("✅ ChromaDB is ready!")
        st.session_state['Vector_store'] = vector_store



PROMPT_TEMPLATE = """
    - Répondez à la question en fonction du contexte suivant :

    {context}

    ---

    - La question est : {question}
    - Si le contexte est hors question, répondez avec vos propres informations.
    - Si la question de l'utilisateur est générale, ne demandez pas de contexte, ignorez le contexte fourni.

"""

def Retrieve(Question):
    db = Chroma(persist_directory=persist_directory, embedding_function=embed_model)
    # Searching for Relevent Chunks from DataBase
    results = db.similarity_search_with_relevance_scores(Question,k=5)
    context_text = "\n\n---\n\n".join([chunk.page_content for chunk, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # The Prompt
    prompt = prompt_template.format(context=context_text, question=Question)
    return prompt,context_text


def Run_Pipeline(question,LLM_Name):
    prompt,_ = Retrieve(question)
    st.write("### 🧾 Prompt")
    st.text_area(label = "", value = prompt, height= 200)

    llm = Ollama(model=LLM_Name,base_url = URL)
    response = llm.invoke(prompt)
    return response

def RunLLM(question,LLM_Name):
    llm = Ollama(model=LLM_Name,base_url = URL)
    response = llm.invoke(question)
    return response
