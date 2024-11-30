import streamlit as st

from utilitis import Proccess_Files,Chunking,Create_Database,Run_Pipeline,RunLLM
from utilitis1 import Proccess_Files_1,Chunking_1,Create_Database_1,Run_Pipeline_1

model_choice = st.sidebar.selectbox(
    "Select a model",
    ["llama3.2:3b", "llama3.1:8b", "qwen:7b"]
)

Check_Box = st.sidebar.checkbox("Chat With your files", value=False)


if Check_Box :
    Arabic_Files = st.sidebar.checkbox("Arabic Docs", value=False)

    if Arabic_Files :

        st.sidebar.title("! قم برفع ملفاتك هنا")
        uploaded_files = st.sidebar.file_uploader("اختر الملفات", type="pdf", accept_multiple_files=True)
        if 'Documents_1' not in st.session_state :
            Documents = Proccess_Files_1(uploaded_files)
        if 'Chunks_1' not in st.session_state :
            Chunks = Chunking_1(Documents)
        if 'Vector_store_1' not in st.session_state :
            Create_Database_1(Chunks)

        if 'Vector_store_1' in st.session_state :
            st.write("### ✨ اسأل ملفاتك...")
            question = st.text_input("")
            if st.button("إرسال"):
                if question:
                    with st.spinner("يفكر..."):
                        response =Run_Pipeline_1(question,model_choice)
                        st.write(f"### 🤖 الرد :({model_choice})")
                        st.write(response)
                else:
                    st.write("### ✨ اسأل ملفاتك...")

    else :
        st.sidebar.title("Upload your files Here !")
        uploaded_files = st.sidebar.file_uploader("Choose files", type="pdf", accept_multiple_files=True)

        if 'Documents' not in st.session_state :
            Documents = Proccess_Files(uploaded_files)
        if 'Chunks' not in st.session_state :
            Chunks = Chunking(Documents)
        if 'Vector_store' not in st.session_state :
            Create_Database(Chunks)

        if 'Vector_store' in st.session_state :
            st.write("### ✨ Ask your files...")
            question = st.text_input("")
            if st.button("Send"):
                if question:
                    with st.spinner("Is Thinking..."):
                        response =Run_Pipeline(question,model_choice)
                        st.write(f"### 🤖 Response :({model_choice})")
                        st.write(response)
                else:
                    st.write("## ✨ Ask your files...")

else :
    st.write("### ✨ Ask your question...")
    question = st.text_input("")
    if st.button("Send"):
        if question:
            with st.spinner("Is Thinking..."):
                response =RunLLM(question,model_choice)
                st.write(f"### 🤖 Response :({model_choice})")
                st.write(response)











