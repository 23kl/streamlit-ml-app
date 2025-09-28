import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Voice + Translation
from elevenlabs.client import ElevenLabs
from elevenlabs import save
from deep_translator import GoogleTranslator
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

# Streamlit UI
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

# 🔊 Voice mapping (replace with your actual voice IDs from ElevenLabs)
voice_map = {
    "English": "21m00Tcm4TlvDq8ikWAM",   # Rachel
    "Hindi": "1qEiC6qsybMkmnNdVMbK",      # e.g., Akash
    "Marathi": "1qEiC6qsybMkmnNdVMbK",  # Indian-accent male/female
    "Gujarati": "1qEiC6qsybMkmnNdVMbK",
    "Tamil": "gqFUMFHCD2nbbcYVtPGB"
}

# Load vector store
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

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Create RAG pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# Function to clean text before TTS
def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[*_#`]", "", text)  # remove markdown chars
    text = text.replace("•", " ").replace("-", " ")
    text = text.replace(":", ": ")
    return text.strip()

# Input method
mode = st.radio("Choose Input Method:", ["Text", "Voice"])
query_text = ""

if mode == "Text":
    query_text = st.text_input("Ask your question about herbs & regulations:")

elif mode == "Voice":
    st.write("🎙️ Speak your query")
    audio_data = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key="recorder")

    if audio_data and "bytes" in audio_data:
        # Save mic recording temporarily (webm)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmpfile:
            tmpfile.write(audio_data["bytes"])
            tmpfile_path = tmpfile.name

        # Convert WebM → WAV
        wav_path = tmpfile_path.replace(".webm", ".wav")
        sound = AudioSegment.from_file(tmpfile_path, format="webm")
        sound.export(wav_path, format="wav")

        # Recognize speech
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        try:
            query_text = recognizer.recognize_google(audio, language=languages[user_lang])
            st.write("🗣️ You said:", query_text)
        except sr.UnknownValueError:
            st.error("❌ Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"❌ Could not request results; {e}")

# Run query
if st.button("Get Answer"):
    if query_text.strip() == "":
        st.warning("Please enter or speak a question.")
    else:
        with st.spinner("Thinking..."):
            # Translate query → English
            query_in_english = GoogleTranslator(source='auto', target='en').translate(query_text)

            # Get answer
            answer = qa.run(query_in_english)

            # Translate back to user language
            answer_translated = GoogleTranslator(source='en', target=languages[user_lang]).translate(answer)

            # Show text
            st.success(answer_translated)

            # Prepare for TTS
            tts_text = clean_text_for_tts(answer_translated)

            # ElevenLabs TTS
            client = ElevenLabs(api_key=elevenlabs_api_key)

            # Pick voice for the chosen language
            selected_voice = voice_map.get(user_lang, "21m00Tcm4TlvDq8ikWAM")  # fallback Rachel

            audio = client.text_to_speech.convert(
                text=tts_text,
                voice_id=selected_voice,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )

            # Save & Play
            audio_file = "response.mp3"
            save(audio, audio_file)
            st.audio(audio_file, format="audio/mp3")

