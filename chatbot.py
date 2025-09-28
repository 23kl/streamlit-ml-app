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
from gtts import gTTS
from deep_translator import GoogleTranslator
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import io

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

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
    text = re.sub(r"[*_#`]", "", text)
    text = text.replace("•", " ").replace("-", " ")
    text = text.replace(":", ": ")
    return text.strip()

# Function to generate audio using gTTS
def text_to_speech_gtts(text: str, lang_code: str):
    try:
        cleaned_text = clean_text_for_tts(text)
        if not cleaned_text.strip():
            return None
        tts = gTTS(text=cleaned_text, lang=lang_code, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

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
            try:
                # Translate query → English
                query_in_english = GoogleTranslator(source='auto', target='en').translate(query_text)

                # Get answer
                answer = qa.run(query_in_english)

                # Translate back to user language
                answer_translated = GoogleTranslator(source='en', target=languages[user_lang]).translate(answer)

                # Show text
                st.subheader("📝 Answer:")
                st.success(answer_translated)

                # Generate audio
                st.subheader("🔊 Audio Response:")
                audio_fp = text_to_speech_gtts(answer_translated, languages[user_lang])
                if audio_fp:
                    st.audio(audio_fp, format="audio/mp3")
                else:
                    st.info("Audio generation failed, please read the text.")

            except Exception as e:
                st.error(f"Error processing your request: {e}")

