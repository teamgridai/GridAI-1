import os
import openai
import pdfplumber
import fitz
import tempfile
import streamlit as st
from typing import List
from dotenv import load_dotenv

openai.api_key = st.secrets["OPENAI_API_KEY"]
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_text_from_pdfs(pdf_paths: List[str]) -> str:
    """
    Extracts and concatenates text from a list of PDF file paths.
    """
    combined_text = ""

    for path in pdf_paths:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    combined_text += text + "\n"
                else:
                    # Fallback: render page to image with PyMuPDF and OCR
                    doc = fitz.open(path)
                    page = doc.load_page(i)  # zero-based page number
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")

                    # Use temp file to create a temp file for pytesseract
                    with tempfile.NamedTemporaryFile(suffix=".png") as temp_img_file:
                        temp_img_file.write(img_bytes)
                        temp_img_file.flush()
                        ocr_text = page.get_text()
                        combined_text += ocr_text + "\n"

    return combined_text


def chunk_text(text: str, chunk_size: int = 3000) -> List[str]:
    """
    Splits text into chunks for use with OpenAI API.
    """
    words = text.split()
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    return chunks


def ask_question_over_chunks(chunks: List[str], chat_history: List[dict]) -> str:
    """
    Uses OpenAI GPT-4 to answer a conversation with memory over multiple chunks of PDF context.
    The latest user message in chat_history will be augmented with the combined PDF context.
    """
    # Combine PDF chunks into one context string (you can limit length if needed)
    context = "\n".join(chunks)

    # Build messages for OpenAI:
    # Start with system prompt
    messages = [
        {"role": "system", "content": "You are Eric, a concise and knowledgeable assistant in substation engineering and program management."}
    ]

    # Add all messages except the latest user message
    for msg in chat_history[:-1]:
        messages.append(msg)

    # Augment the latest user message by adding the full PDF context
    latest_user_msg = chat_history[-1]
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\n{latest_user_msg['content']}"
    })

    # Call OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error from OpenAI: {e}]"


def ask_general_question(chat_history: List[dict]) -> str:
    system_prompt = """
    You are Eric, a concise and knowledgeable assistant in substation engineering and program management.
    - If the user asks a technical question (like standards or best practices), give a direct and accurate answer.
    - If they ask who you are, respond with: "My name is Eric, your engineering assistant."
    - If the question is outside your expertise, say so politely.
    - Avoid filler, keep responses concise.
    - Never say you're an AI.
    - Do not guess or make up standard numbers.
    - Always stay on topic.
    
    Use the following reference knowledge when answering questions:
    - IEEE Std 80 is the standard for substation grounding design.
    - Transmission substations operate at 115kV to 765kV.
    - IEEE Std 142 covers grounding of industrial and commercial power systems.
    - NESC governs safety standards for electric supply and communication lines.
    - pdf files are uploadable and can be read by you
    - you can review documents directly. Answer specific questions about your electrical grid plans, feel free to ask, and I'll do my best to assist you.
    """

    messages = [{"role": "system", "content": system_prompt}]

    messages.extend(chat_history)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error from OpenAI: {e}]"
