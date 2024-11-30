🧐 Chat with your CV and Cove letter
=================================

Introduction
------------
The notebook provides a step-by-step guide to building a simple pdf-RAG system. It involves:

- Using your **CV** and **Cove Letter** as the knowledge base for RAG operations.
- Leveraging **LangChain** for retrieval and generation tasks.
- Employing **Ollama** as the local large language model backend.

Dependencies
------------
The required libraries are installed using the following commands:

.. code-block:: python

    pip install -qU langchain langchain_community
    pip install -qU langchain_chroma
    pip install -qU langchain_ollama
    pip install pypdf

Additionally, if running in environments like Google Colab, the notebook includes special setups for using **Ollama**.

Notebook Overview
-----------------

1. **Setting Up the Environment**:

   - Installing dependencies.
   - Initializing the tools for retrieval and embedding.

2. **Loading the Pdfs**:

   - In the notebook I include the github link to the my PDFs, but feel free to use your own.

3. **Building the RAG Pipeline**:

   - **Chunking**: Breaks the PDFs into smaller chunks for embedding.
   - **Create ChromaDB**: Creates a vector database for storing the embeddings.
   - **Clone a prompt**:  we use a predifined prompt to interact with our RAG pipeline, but you can create your own.
   - **Retrieve**: Retrieves relevant chunks using semantic search.
   - **Generate**: Combines the retrieved context with the query and sends it to the LLM.

4. **Testing the System**:
   - Users can input queries to test how the RAG pipeline responds.
   - The outputs are evaluated for relevance and accuracy.


.. raw:: html

   <a href="https://colab.research.google.com/github/ITSAIDI/RAGify/blob/main/Notebooks/RAG_2.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Streamlit Interface
-------------------

You can build a simple python interface with Streamlit library to interact with your RAG system.

.. code-block:: python

    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    import streamlit as st
    from langchain.vectorstores import Chroma
    from langchain.chains import create_retrieval_chain
    from langchain import hub
    from langchain.chains.combine_documents import create_stuff_documents_chain

.. hint::

   - You need to save your ChromaDB folder in your machine, or like I did in Github. So you don't need to create it again.

.. code-block:: python

    # The Chroma is On Github     https://github.com/NourTechNerd/data
    persist_directory ="CHROMA"

    embed_model = OllamaEmbeddings(model="llama3.1:8b",base_url="http://127.0.0.1:11434")
    vector_store = Chroma(persist_directory=persist_directory,embedding_function=embed_model)
    llm = Ollama(model="llama3.1:8b",base_url="http://127.0.0.1:11434")
    retriever = vector_store.as_retriever(search_kwargs={"k":3})
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combined_docs_chain = create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever,combined_docs_chain)

1. **Chroma Vector Store**

   - `persist_directory`: Specifies the directory where ChromaDB will persist data.
   - `vector_store`: Initializes a Chroma vector database with the embedding function.

2. **Embedding Model**

   - `OllamaEmbeddings`: A model to compute vector representations of text using the `llama3.1:8b` model hosted locally at `127.0.0.1:11434`.

3. **Local Language Model (LLM)**

   - `llm`: Instantiates the `llama3.1:8b` model for generating text and answering queries.

4. **Retriever**

   - `retriever`: Configures the vector store to return the top 3 (`k=3`) most relevant documents for a query.

5. **Prompt and Chain**

   - `retrieval_qa_chat_prompt`: Fetches a pre-defined prompt template for retrieval-based Q&A tasks.
   - `combined_docs_chain`: Combines the retrieval system with the LLM for document-based answers.
   - `retrieval_chain`: Creates the full pipeline that integrates retrieval and generation.

.. code-block:: python

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


Demo Video
----------
Here is a video of the RAG pipeline in action:

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/SUeJpD8zP1o" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

