import os
import re
import io
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Voice + Translation
from deep_translator import GoogleTranslator
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import tempfile
import base64

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Farmer's Herb Chatbot", page_icon="🌿")
st.title("🌿 Farmer’s Herb & Regulation Chatbot")

# Language selector
languages = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Tamil": "ta"
}
user_lang = st.selectbox("🌐 Select your language:", list(languages.keys()), index=0)

# ---------------------------
# Load vector store
# ---------------------------
@st.cache_resource
def load_vector_store():
    loader = TextLoader("app/india_herbs_regions_soil_climate_rules.csv")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_vector_store()

# ---------------------------
# Initialize Groq LLM
# ---------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# RAG pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# ---------------------------
# Utility functions
# ---------------------------
def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[*_#`]", "", text)  # remove markdown
    text = text.replace("•", " ").replace("-", " ")
    text = text.replace(":", ": ")
    return text.strip()

def text_to_audio_stream(text: str, lang_code: str):
    """Generate base64-encoded audio using gTTS for Streamlit playback."""
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang_code, slow=False)
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmpfile.name)
    return tmpfile.name

# ---------------------------
# Input method
# ---------------------------
mode = st.radio("Choose Input Method:", ["Text", "Voice"])
query_text = ""

# --- Text Input ---
if mode == "Text":
    query_text = st.text_input("Ask your question about herbs & regulations:")

# --- Voice Input ---
elif mode == "Voice":
    st.write("🎙️ Speak your query")
    audio_data = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key="recorder")

    if audio_data and "bytes" in audio_data and audio_data["bytes"]:
        try:
            # Convert WebM bytes to AudioSegment
            webm_fp = io.BytesIO(audio_data["bytes"])
            sound = AudioSegment.from_file(webm_fp, format="webm")

            # Export to WAV in-memory
            wav_fp = io.BytesIO()
            sound.export(wav_fp, format="wav")
            wav_fp.seek(0)

            # Recognize speech
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_fp) as source:
                audio = recognizer.record(source)
            query_text = recognizer.recognize_google(audio, language=languages[user_lang])
            st.write("🗣️ You said:", query_text)
        except sr.UnknownValueError:
            st.error("❌ Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"❌ Speech recognition error: {e}")
        except Exception as e:
            st.error(f"❌ Audio processing failed: {e}")

# ---------------------------
# Run query
# ---------------------------
if st.button("Get Answer"):
    if not query_text.strip():
        st.warning("Please enter or speak a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Translate query → English
                query_in_english = GoogleTranslator(source='auto', target='en').translate(query_text)

                # Get answer from Groq
                answer = qa.run(query_in_english)

                # Translate back to user language
                answer_translated = GoogleTranslator(source='en', target=languages[user_lang]).translate(answer)

                # Display text
                st.subheader("📝 Answer:")
                st.success(answer_translated)

                # Generate audio
                audio_file = text_to_audio_stream(answer_translated, languages[user_lang])

                # Play audio
                st.subheader("🔊 Audio Response:")
                st.audio(audio_file, format="audio/mp3")
        except Exception as e:
            st.error(f"❌ Error processing your request: {e}")



