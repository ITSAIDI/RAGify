
Chat with your CV and Cove letter
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


Demo Video
----------
Here is a video of the RAG pipeline in action:

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/SUeJpD8zP1o" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

