from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_core.documents import Document
import streamlit as st
import chromadb
from langchain.prompts import ChatPromptTemplate

chromadb.api.client.SharedSystemClient.clear_system_cache()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
URL = "http://localhost:11434"
persist_directory="CHROMA_Arabic"

def Extract_pdf_content_1(pdf_file):
    """
    Extracts the content of each page in a PDF file and returns a list of pages.
    """
    reader = PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text())
    return pages

def Proccess_Files_1(Files):
    if Files : 
        st.title("📄 قراءة الملفات ...")
        progress_percentage = 0
        Documents = []
        
        # Initialize the progress bar
        total_files = len(Files)
        progress_bar = st.progress(0)
        
        for file_index, file in enumerate(Files):
            # Extract content from the PDF
            Pages_Contents = Extract_pdf_content_1(file)
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
            st.success("✅ تم الانتهاء من معالجة الملفات!")
            st.session_state['Documents_1'] = Documents

        print(Documents)
        return Documents


def Chunking_1(documents):
    if documents :
        st.title("✂️ تقسيم الوثائق ...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=600)
        Chunks = text_splitter.split_documents(documents)
        st.write("#### Number of Chunks is :",len(Chunks))
        if Chunks :
            st.success("✅ تم تقسيم الوثائق بنجاح!")
            st.session_state['Chunks_1'] = Chunks
        return Chunks
    return None

def Create_Database_1(Chunks):
    if Chunks :
       st.title("🗄️ إنشاء قاعدة بيانات ChromaDB ...")
       vector_store = Chroma.from_documents(Chunks,embedding_model,persist_directory=persist_directory)
       st.success("✅ قاعدة بيانات  جاهزة!")
       st.session_state['Vector_store_1'] = vector_store



PROMPT_TEMPLATE = """
    - أجب على السؤال بناءً على السياق التالي :

    {context}

    ---

    - السؤال هو : {question}
    - إذا كان السياق لا يتعلق بالسؤال، أجب بمعلوماتك الخاصة.
    - إذا كان السؤال العام من المستخدم، لا تطلب سياقًا، تجاهل السياق المقدم.

"""

def Retrieve_1(Question):
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    # Searching for Relevent Chunks from DataBase
    results = db.similarity_search_with_relevance_scores(Question,k=5)
    context_text = "\n\n---\n\n".join([chunk.page_content for chunk, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # The Prompt
    prompt = prompt_template.format(context=context_text, question=Question)
    return prompt,context_text


def Run_Pipeline_1(question,LLM_Name):
    prompt,_ = Retrieve_1(question)
    st.write("### 🧾 الطلب")
    st.text_area(label = "", value = prompt, height= 200)

    llm = Ollama(model=LLM_Name,base_url = URL)
    response = llm.invoke(prompt)
    return response













