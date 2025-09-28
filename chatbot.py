import os
import re
import io
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Voice + Translation
from gtts import gTTS
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

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
        possible_paths = [
            "india_herbs_regions_soil_climate_rules.csv",
            "app/india_herbs_regions_soil_climate_rules.csv",
            "data/india_herbs_regions_soil_climate_rules.csv",
            "./india_herbs_regions_soil_climate_rules.csv"
        ]
        file_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if not file_path:
            st.error("❌ Data file not found.")
            return None

        # Load CSV first, fallback to text
        try:
            loader = CSVLoader(file_path=file_path)
            documents = loader.load()
        except:
            loader = TextLoader(file_path=file_path)
            documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore

    except Exception as e:
        st.error(f"❌ Error loading documents: {str(e)}")
        return None

# Initialize RAG pipeline
vectorstore = load_vector_store()
if vectorstore is None:
    st.stop()

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

# Clean text for TTS
def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[*_#`]", "", text)
    text = text.replace("•", " ").replace("-", " ").replace(":", ": ")
    return text.strip()

# Convert text to in-memory audio
def text_to_speech_base64(text, lang_code="en"):
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
        st.error(f"TTS Error: {e}")
        return None

# Browser-based voice recording using WebRTC
def record_audio_webrtc():
    webrtc_ctx = webrtc_streamer(
        key="voice-input",
        mode=WebRtcMode.RECVONLY,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False}
        ),
        async_processing=False
    )
    if webrtc_ctx.audio_receiver:
        frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        if frames:
            audio_bytes = b"".join([f.to_bytes() for f in frames])
            audio_segment = AudioSegment.from_raw(io.BytesIO(audio_bytes), sample_width=2, frame_rate=44100, channels=1)
            audio_fp = io.BytesIO()
            audio_segment.export(audio_fp, format="wav")
            audio_fp.seek(0)
            return audio_fp
    return None

# Transcribe audio using Google Speech Recognition
def transcribe_audio(audio_fp, language):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_fp) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition error: {e}")
        return None

# Input mode
mode = st.radio("Choose Input Method:", ["Text", "Voice"])
query_text = ""

if mode == "Text":
    query_text = st.text_input("Ask your question about herbs & regulations:")

elif mode == "Voice":
    st.write("🎤 Voice Input")
    if st.button("🎙️ Start Recording", key="record_btn_webrtc"):
        with st.spinner("Recording... Speak now!"):
            audio_fp = record_audio_webrtc()
        if audio_fp:
            query_text = transcribe_audio(audio_fp, languages[user_lang])
            if query_text:
                st.success(f"🗣️ You said: **{query_text}**")

# Run query
if st.button("Get Answer", key="get_answer"):
    if not query_text.strip():
        st.warning("Please enter or speak a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                st.write(f"**Your question:** {query_text}")

                # Translate query → English
                if user_lang != "English":
                    query_in_english = GoogleTranslator(source='auto', target='en').translate(query_text)
                    st.write(f"**Translated question:** {query_in_english}")
                else:
                    query_in_english = query_text

                # Get answer
                answer = qa.run(query_in_english)

                # Translate answer back
                if user_lang != "English":
                    answer_translated = GoogleTranslator(source='en', target=languages[user_lang]).translate(answer)
                else:
                    answer_translated = answer

                # Show answer
                st.subheader("📝 Answer:")
                st.success(answer_translated)

                # Generate audio response
                st.subheader("🔊 Audio Response:")
                audio_fp = text_to_speech_base64(answer_translated, languages[user_lang])
                if audio_fp:
                    st.audio(audio_fp, format="audio/mp3")
                else:
                    st.info("Audio generation failed, please read the text.")

            except Exception as e:
                st.error(f"Error processing your request: {e}")

# Sidebar info
st.sidebar.markdown("""
### 💡 How to use:
1. Select language
2. Choose input method: Text or Voice
3. Ask questions about:
   - Herb cultivation
   - Regional suitability
   - Soil requirements
   - Climate conditions
   - Government regulations
   - Farming techniques

### 🎤 Voice Input Tips:
- Click "Start Recording"
- Speak clearly in quiet environment
- Allow microphone permissions

### 🌿 Supported Languages:
- English, Hindi, Marathi, Gujarati, Tamil
""")

