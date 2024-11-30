RAG example 
===========

This notebook demonstrates the creation and testing of a Retrieval-Augmented Generation (RAG) system using **LangChain** and **Ollama**. Below is a detailed breakdown of its structure and purpose.

Introduction
------------
The notebook provides a step-by-step guide to building a simple RAG system. It involves:

- Using a small corpus as the knowledge base for RAG operations.
- Leveraging **LangChain** for retrieval and generation tasks.
- Employing **Ollama** as the local large language model backend.

Dependencies
------------
The required libraries are installed using the following commands:

.. code-block:: python

    !pip install -qU langchain langchain_community
    !pip install -qU langchain_chroma
    !pip install -qU langchain_ollama

Additionally, if running in environments like Google Colab, the notebook includes special setups for using **Ollama**.

Notebook Overview
-----------------

1. **Setting Up the Environment**:
   - Installing dependencies.
   - Initializing the tools for retrieval and embedding.

2. **Loading the Corpus**:
   - The notebook uses a small text-dataset as the source of knowledge.
   - It preprocesses the corpus into chunks for embedding.

3. **Building the RAG Pipeline**:
   - **Retrieve**: Retrieves relevant chunks using semantic search.
   - **Generate**: Combines the retrieved context with the query and sends it to the LLM.

4. **Testing the System**:
   - Users can input queries to test how the RAG pipeline responds.
   - The outputs are evaluated for relevance and accuracy.


.. raw:: html

   <a href="https://colab.research.google.com/github/ITSAIDI/RAGify/blob/main/Notebooks/RAG_1.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
