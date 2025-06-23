import streamlit as st
from GridML import (
    extract_text_from_pdfs,
    chunk_text,
    ask_question_over_chunks,
    ask_general_question
)
import tempfile
import os

def main():
    st.title("GridAI")

    # Upload PDFs but optional
    uploaded_files = st.file_uploader(
        "Upload your PDF files (optional)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            temp_dir = tempfile.mkdtemp()
            pdf_paths = []
            for file in uploaded_files:
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                pdf_paths.append(temp_path)
            full_text = extract_text_from_pdfs(pdf_paths)
            st.session_state["pdf_text_chunks"] = chunk_text(full_text)
            st.success(f"Loaded {len(uploaded_files)} PDFs.")

    question = st.text_input("Enter your question:")

    if question:
        with st.spinner("Getting answer..."):
            try:
                if "pdf_text_chunks" in st.session_state:
                    # Use PDF context
                    answer = ask_question_over_chunks(st.session_state["pdf_text_chunks"], question)
                else:
                    # Use general GPT-4 prompt without PDF context
                    answer = ask_general_question(question)
                st.markdown("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Failed to get answer: {e}")
if __name__ == "__main__":
    main()
