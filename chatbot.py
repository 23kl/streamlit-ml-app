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
import base64

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Farmer's Herb Chatbot", page_icon="🌿")
st.title("🌿 Farmer's Herb & Regulation Chatbot")

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
        loader = TextLoader("india_herbs_regions_soil_climate_rules.csv")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return None

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

# Function to create audio file and return base64 encoded audio
def text_to_speech(text, lang_code):
    try:
        # Clean text
        cleaned_text = clean_text_for_tts(text)
        
        # Create gTTS object
        tts = gTTS(text=cleaned_text, lang=lang_code, slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts.save(tmpfile.name)
            return tmpfile.name
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return None

# Function to autoplay audio
def autoplay_audio(audio_file):
    try:
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay controls style="width: 100%">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error playing audio: {e}")

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
        try:
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
        except Exception as e:
            st.error(f"Error processing audio: {e}")
        finally:
            # Clean up temporary files
            if os.path.exists(tmpfile_path):
                os.unlink(tmpfile_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)

# Run query
if st.button("Get Answer"):
    if query_text.strip() == "":
        st.warning("Please enter or speak a question.")
    elif vectorstore is None:
        st.error("Document database not loaded. Please check your data file.")
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
                st.success(answer_translated)

                # Generate and play audio
                audio_file = text_to_speech(answer_translated, languages[user_lang])
                if audio_file:
                    st.write("🔊 Audio Response:")
                    autoplay_audio(audio_file)
                    
                    # Clean up audio file
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
                        
            except Exception as e:
                st.error(f"Error processing your request: {e}")

