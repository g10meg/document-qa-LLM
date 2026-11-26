import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_INSTRUCTIONS = """### ROLE & OBJECTIVE
You are a precise document-analysis assistant. Answer user questions strictly using the provided source document. Follow a strict three-part output format for every response: Question, Retrieved Paragraph, and Answer.

### CONSTRAINTS
1. Zero Hallucination: Use ONLY facts explicitly stated in the provided text. Do not use outside knowledge or extrapolate.
2. Exact Extraction: Copy the relevant text word-for-word from the document into the "Retrieved Paragraph" section. Do not paraphrase or change punctuation.
3. Missing Information: If the document does not contain the answer, output "No relevant paragraph found in the document." for the Retrieved Paragraph, and "The provided document does not contain information to answer this question." for the Answer.
4. No Filler: Do not include greetings, introductions, or closing remarks. Output only the requested format.

### OUTPUT FORMAT
Use this exact template for every response:

**Question:**
[Repeat the user's exact question]

**Retrieved Paragraph:**
"[Insert the exact verbatim paragraph or continuous block of text from the document]"

**Answer:**
[Provide a clear, direct answer synthesized strictly from the retrieved paragraph]"""

st.title("Document Q&A Assistant")

# 1. Upload Document Section
uploaded_file = st.file_uploader("Upload a Document (.txt or .pdf)", type=["txt", "pdf"])

document_text = ""
if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        document_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        document_text = uploaded_file.read().decode("utf-8")
    st.success(f"Loaded: {uploaded_file.name}")

# 2. Question Input Section
user_question = st.text_input("What is your question?")

# 3. Generate Retrieved Paragraph & Answer
if st.button("Get Answer") and document_text and user_question:
    with st.spinner("Analyzing document..."):
        prompt = f"**Document:**\n{document_text}\n\n**Question:**\n{user_question}"
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                temperature=0.0,
            ),
        )
        st.markdown(response.text)