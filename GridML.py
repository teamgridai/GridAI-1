import os
import pdfplumber
import openai
from typing import List
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_text_from_pdfs(pdf_paths: List[str]) -> str:
    """
    Extracts and concatenates text from a list of PDF file paths.
    """
    combined_text = ""
    for path in pdf_paths:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    combined_text += page_text + "\n"
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


def ask_question_over_chunks(chunks: List[str], question: str) -> str:
    """
    Uses OpenAI GPT-4 to answer a question over multiple chunks of context.
    """
    answers = []
    for chunk in chunks:
        messages = [
            {"role": "system", "content": "You are Eric, a concise and knowledgeable assistant in substation engineering and program management."},
            {"role": "user", "content": f"Context:\n{chunk}\n\nQuestion: {question}"}
        ]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
                max_tokens=512
            )
            answer = response["choices"][0]["message"]["content"]
            answers.append(answer)
        except Exception as e:
            answers.append(f"[Error from OpenAI: {e}]")
    return "\n\n".join(answers)

def ask_general_question(question: str) -> str:
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
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        max_tokens=512
    )
    return response.choices[0].message.content