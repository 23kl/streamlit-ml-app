import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Voice + Translation
from gtts import gTTS
from deep_translator import GoogleTranslator
import speech_recognition as sr
import base64
import soundfile as sf

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
    try:
        file_path = None
        possible_paths = [
            "india_herbs_regions_soil_climate_rules.csv",
            "app/india_herbs_regions_soil_climate_rules.csv",
            "data/india_herbs_regions_soil_climate_rules.csv"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        if not file_path:
            st.error("❌ CSV file not found. Please add 'india_herbs_regions_soil_climate_rules.csv'.")
            return None

        # Load documents
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Create vector store
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error loading documents: {e}")
        return None

vectorstore = load_vector_store()
if vectorstore is None:
    st.stop()

# Initialize Groq LLM
try:
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )
except Exception as e:
    st.error(f"❌ Error initializing Groq model: {e}")
    st.stop()

# Function to clean text for TTS
def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[*_#`]", "", text)
    text = text.replace("•", " ").replace("-", " ")
    text = text.replace(":", ": ")
    return text.strip()

# Function to generate audio
def text_to_speech(text, lang_code):
    try:
        cleaned_text = clean_text_for_tts(text)
        if not cleaned_text.strip():
            return None
        tts = gTTS(text=cleaned_text, lang=lang_code, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts.save(tmpfile.name)
            return tmpfile.name
    except Exception as e:
        st.error(f"❌ Error in TTS: {e}")
        return None

# Function to play audio
def autoplay_audio(audio_file):
    try:
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay controls style="width:100%">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error playing audio: {e}")

# Input method
mode = st.radio("Choose Input Method:", ["Text", "Voice"])
query_text = ""

if mode == "Text":
    query_text = st.text_input("Ask your question about herbs & regulations:")

elif mode == "Voice":
    st.write("🎤 Click 'Record' and speak your question")
    if st.button("🎙️ Record"):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                st.info("Listening...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            query_text = recognizer.recognize_google(audio, language=languages[user_lang])
            st.success(f"🗣️ You said: {query_text}")
        except sr.WaitTimeoutError:
            st.error("⏰ No speech detected. Please try again.")
        except sr.UnknownValueError:
            st.error("❌ Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"❌ Speech recognition error: {e}")
        except Exception as e:
            st.error(f"❌ Microphone error: {e}")

# Run query
if st.button("Get Answer"):
    if not query_text.strip():
        st.warning("Please enter or speak a question.")
    else:
        try:
            with st.spinner("Processing your question..."):
                # Translate to English
                if user_lang != "English":
                    query_in_english = GoogleTranslator(source='auto', target='en').translate(query_text)
                else:
                    query_in_english = query_text

                # Get answer from Groq
                answer = qa.run(query_in_english)

                # Translate back
                if user_lang != "English":
                    answer_translated = GoogleTranslator(source='en', target=languages[user_lang]).translate(answer)
                else:
                    answer_translated = answer

                st.subheader("📝 Answer:")
                st.success(answer_translated)

                st.subheader("🔊 Audio Response:")
                audio_file = text_to_speech(answer_translated, languages[user_lang])
                if audio_file:
                    autoplay_audio(audio_file)
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
                else:
                    st.info("Audio generation failed.")
        except Exception as e:
            st.error(f"❌ Error processing your request: {e}")


