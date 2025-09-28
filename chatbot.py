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
                st.success(f"Found data file at: {path}")
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
        st.success("✅ Document database loaded successfully!")
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

# Improved audio processing function
def process_audio_data(audio_data):
    """Process audio data from streamlit-mic-recorder with better error handling"""
    if not audio_data or "bytes" not in audio_data:
        return None
        
    try:
        # Save the raw bytes to a file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            # Try to write as WAV directly if possible
            tmpfile.write(audio_data["bytes"])
            tmpfile_path = tmpfile.name
        
        # Try different approaches to read the audio
        audio = None
        
        # Approach 1: Try reading as WAV directly
        try:
            audio = AudioSegment.from_file(tmpfile_path, format="wav")
        except:
            # Approach 2: Try reading as raw data with parameters
            try:
                audio = AudioSegment(
                    data=audio_data["bytes"],
                    sample_width=2,
                    frame_rate=44100,
                    channels=1
                )
            except:
                # Approach 3: Use alternative conversion
                try:
                    # Save as raw and let pydub detect format
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as raw_file:
                        raw_file.write(audio_data["bytes"])
                        raw_path = raw_file.name
                    
                    audio = AudioSegment.from_file(raw_path)
                    os.unlink(raw_path)
                except Exception as e:
                    st.error(f"All audio conversion methods failed: {e}")
                    return None
        
        # Export to WAV for speech recognition
        wav_path = tmpfile_path.replace(".wav", "_processed.wav")
        audio.export(wav_path, format="wav")
        
        return wav_path
        
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None
    finally:
        # Clean up temporary files
        try:
            if 'tmpfile_path' in locals() and os.path.exists(tmpfile_path):
                os.unlink(tmpfile_path)
        except:
            pass

# Input method
mode = st.radio("Choose Input Method:", ["Text", "Voice"])
query_text = ""

if mode == "Text":
    query_text = st.text_input("Ask your question about herbs & regulations:")

elif mode == "Voice":
    st.write("🎙️ Speak your query (speak clearly after clicking 'Start Recording')")
    
    # Use mic_recorder with WAV format if possible
    audio_data = mic_recorder(
        start_prompt="🎤 Start Recording", 
        stop_prompt="⏹️ Stop Recording", 
        key="recorder",
        format="wav" if hasattr(mic_recorder, 'format') else None
    )

    if audio_data:
        with st.spinner("Processing audio..."):
            processed_audio_path = process_audio_data(audio_data)
            
            if processed_audio_path:
                try:
                    # Recognize speech
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(processed_audio_path) as source:
                        # Adjust for ambient noise
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = recognizer.record(source)
                    
                    try:
                        query_text = recognizer.recognize_google(audio, language=languages[user_lang])
                        st.success(f"🗣️ You said: {query_text}")
                    except sr.UnknownValueError:
                        st.error("❌ Sorry, I could not understand the audio. Please try again.")
                    except sr.RequestError as e:
                        st.error(f"❌ Speech recognition error: {e}")
                        
                except Exception as e:
                    st.error(f"❌ Error processing audio: {e}")
                finally:
                    # Clean up processed audio file
                    try:
                        if os.path.exists(processed_audio_path):
                            os.unlink(processed_audio_path)
                    except:
                        pass
            else:
                st.error("❌ Failed to process audio recording. Please try again.")

# Alternative simple voice input using speech_recognition directly
st.markdown("---")
st.subheader("Alternative Voice Input")
if st.button("🎤 Use Simple Voice Input"):
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Speak now...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10)
            
        query_text = recognizer.recognize_google(audio, language=languages[user_lang])
        st.success(f"🗣️ You said: {query_text}")
    except sr.WaitTimeoutError:
        st.error("❌ No speech detected within 10 seconds.")
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio.")
    except Exception as e:
        st.error(f"❌ Voice input error: {e}")

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
                st.write(answer_translated)

                # Generate and play audio
                st.subheader("🔊 Audio Response:")
                audio_file = text_to_speech(answer_translated, languages[user_lang])
                if audio_file:
                    autoplay_audio(audio_file)
                    
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
- Speak clearly and slowly
- Use the alternative voice input if the main one fails
- Ensure microphone permissions are granted
- Use in a quiet environment

### 🌿 Supported Languages:
- English
- Hindi  
- Marathi
- Gujarati
- Tamil
""")
