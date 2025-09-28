import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Voice + Translation
from gtts import gTTS
from deep_translator import GoogleTranslator
import speech_recognition as sr
import base64
import io

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

# Load vector store with better error handling
@st.cache_resource
def load_vector_store():
    try:
        # Try different possible file paths and formats
        possible_paths = [
            "india_herbs_regions_soil_climate_rules.csv",
            "app/india_herbs_regions_soil_climate_rules.csv", 
            "data/india_herbs_regions_soil_climate_rules.csv",
            "./india_herbs_regions_soil_climate_rules.csv"
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            st.error("❌ Data file not found. Please ensure 'india_herbs_regions_soil_climate_rules.csv' is in your project directory.")
            return None
        
        # Try CSVLoader first (more appropriate for CSV files)
        try:
            loader = CSVLoader(file_path=file_path)
            documents = loader.load()
        except:
            # Fallback to TextLoader
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

# Initialize components with error handling
try:
    vectorstore = load_vector_store()
    
    if vectorstore is None:
        st.error("Failed to initialize document database. The app cannot function without data.")
        st.stop()
    
    # Initialize Groq LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

    # Create RAG pipeline
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )
    
except Exception as e:
    st.error(f"❌ Error initializing the application: {str(e)}")
    st.stop()

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
        
        if not cleaned_text.strip():
            return None
            
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

# SIMPLE VOICE INPUT USING speech_recognition
def record_audio():
    """Record audio using speech_recognition"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # Listen for audio with timeout
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
        return audio
    except sr.WaitTimeoutError:
        st.error("⏰ No speech detected. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Microphone error: {e}")
        return None

def transcribe_audio(audio, language):
    """Transcribe audio to text"""
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio. Please speak clearly.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition error: {e}")
        return None

# Input method
mode = st.radio("Choose Input Method:", ["Text", "Voice"])
query_text = ""

if mode == "Text":
    query_text = st.text_input("Ask your question about herbs & regulations:")

elif mode == "Voice":
    st.write("🎤 Voice Input")
    
    # Simple record button approach
    if st.button("🎙️ Start Recording", key="record_btn"):
        with st.spinner("Recording... Speak now!"):
            audio = record_audio()
            
        if audio:
            with st.spinner("Processing your speech..."):
                transcribed_text = transcribe_audio(audio, languages[user_lang])
                
            if transcribed_text:
                query_text = transcribed_text
                st.success(f"🗣️ You said: **{query_text}**")
                
    # Also show current query text if any
    if query_text:
        st.text_input("Current query:", value=query_text, key="voice_query", disabled=True)

# Run query
if st.button("Get Answer", key="get_answer"):
    if not query_text.strip():
        st.warning("Please enter or speak a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Show original query
                st.write(f"**Your question:** {query_text}")
                
                # Translate query → English if not already in English
                if user_lang != "English":
                    with st.spinner("Translating question..."):
                        query_in_english = GoogleTranslator(source='auto', target='en').translate(query_text)
                    st.write(f"**Translated question:** {query_in_english}")
                else:
                    query_in_english = query_text

                # Get answer
                with st.spinner("Searching for answer..."):
                    answer = qa.run(query_in_english)

                # Translate back to user language if not English
                if user_lang != "English":
                    with st.spinner("Translating answer..."):
                        answer_translated = GoogleTranslator(source='en', target=languages[user_lang]).translate(answer)
                else:
                    answer_translated = answer

                # Show text answer
                st.subheader("📝 Answer:")
                st.success(answer_translated)

                # Generate and play audio
                st.subheader("🔊 Audio Response:")
                with st.spinner("Generating audio..."):
                    audio_file = text_to_speech(answer_translated, languages[user_lang])
                    
                if audio_file:
                    autoplay_audio(audio_file)
                    st.info("🎧 Audio is playing...")
                    
                    # Clean up audio file
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
                else:
                    st.info("Audio generation failed, but you can read the text answer above.")
                        
            except Exception as e:
                st.error(f"Error processing your request: {e}")

# Add some helpful information
st.sidebar.markdown("""
### 💡 How to use:
1. **Select your preferred language**
2. **Choose input method**: Text or Voice
3. **Ask questions about**:
   - Herb cultivation
   - Regional suitability  
   - Soil requirements
   - Climate conditions
   - Government regulations
   - Farming techniques

### 🎤 Voice Input Tips:
- Click "Start Recording" and speak clearly
- Wait for the "Listening..." message
- Speak in a quiet environment
- Allow microphone permissions in your browser

### 🌿 Supported Languages:
- English
- Hindi  
- Marathi
- Gujarati
- Tamil

### 🔧 Troubleshooting:
- If voice doesn't work, use text input
- Ensure microphone is connected and enabled
- Speak clearly and not too fast
"""
)

# Display current status
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Language:** {user_lang}")
st.sidebar.markdown(f"**Input Mode:** {mode}")
if query_text:
    st.sidebar.markdown(f"**Current Query:** {query_text[:50]}..." if len(query_text) > 50 else f"**Current Query:** {query_text}")
