import os
import re
import tempfile
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Voice + Translation
from deep_translator import GoogleTranslator
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI Configuration
st.set_page_config(
    page_title="Kisaan Mitra - Farmer's Herb Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Farmer-Friendly Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px #d4f7d4;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        text-align: center;
        margin-bottom: 2rem;
    }
    .farmer-card {
        background: linear-gradient(135deg, #f8fff8, #e8f5e8);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #2E8B57;
        margin: 10px 0px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sidebar-section {
        background: #f0fff0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0px;
        border: 2px solid #90EE90;
    }
    .language-selector {
        background: #e8f5e8;
        padding: 15px;
        border-radius: 10px;
        border: 2px dashed #2E8B57;
    }
    .voice-section {
        background: #f0f8ff;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #87CEEB;
    }
    .answer-box {
        background: linear-gradient(135deg, #e6ffe6, #ccffcc);
        padding: 25px;
        border-radius: 15px;
        border: 3px solid #32CD32;
        margin-top: 20px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stButton>button {
        background: linear-gradient(45deg, #2E8B57, #32CD32);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(46, 139, 87, 0.4);
    }
    .simple-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        margin: 10px 0px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .language-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .input-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .option-title {
        font-size: 1.3rem;
        color: #2E8B57;
        font-weight: bold;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-header">🌿 किसान मित्र - Farmer\'s Herb Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your Digital Companion for Herb Farming & Regulations</div>', unsafe_allow_html=True)

# Sidebar - User Manual
with st.sidebar:
    st.markdown("""
    <div style='text-align: center;'>
        <h2>📚 User Manual</h2>
        <hr style='border: 2px solid #2E8B57;'>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🎯 How to Use")
    st.markdown("""
    1. **Select Your Language** - Choose from 5 Indian languages
    2. **Choose Input Method** - Type or Speak your question
    3. **Ask Your Question** - Get instant answers about:
       - 🌱 Herb cultivation techniques
       - 🗺️ Regional suitability
       - 🌧️ Climate requirements  
       - 🌱 Soil preparation
       - 📜 Government regulations
       - 💰 Market information
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🎤 Voice Input Guide")
    st.markdown("""
    - Click **Start Recording** button
    - Speak clearly in your chosen language
    - Wait for **Stop Recording** to appear
    - Your speech will be converted to text automatically
    - Perfect for farmers who prefer speaking over typing
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🌿 Supported Herbs & Topics")
    st.markdown("""
    - **Medicinal Herbs**: Ashwagandha, Tulsi, Aloe Vera
    - **Aromatic Herbs**: Mint, Lemongrass, Basil
    - **Spice Herbs**: Turmeric, Ginger, Cumin
    - **Regulations**: Organic certification, Export rules
    - **Techniques**: Organic farming, Irrigation methods
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("📞 Need Help?")
    st.markdown("""
    - Check microphone permissions for voice
    - Speak clearly in a quiet environment
    - For technical issues, contact support
    - Always verify critical information with local experts
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Content Area - Simplified Selection Section
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="simple-card">', unsafe_allow_html=True)
    st.markdown('<div class="language-icon">🌐</div>', unsafe_allow_html=True)
    st.markdown('<div class="option-title">Choose Your Language</div>', unsafe_allow_html=True)
    
    languages = {
        "English": "en",
        "Hindi": "hi",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Tamil": "ta"
    }
    user_lang = st.selectbox(
        "Select your preferred language:",
        list(languages.keys()),
        index=0,
        label_visibility="collapsed"
    )
    st.markdown(f"**Selected:** {user_lang} ✅")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="simple-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-icon">💬</div>', unsafe_allow_html=True)
    st.markdown('<div class="option-title">Choose Input Method</div>', unsafe_allow_html=True)
    
    mode = st.radio(
        "How would you like to ask your question?",
        ["Text", "Voice"],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown(f"**Selected:** {mode} ✅")
    st.markdown('</div>', unsafe_allow_html=True)

# Question Input Section
st.markdown("### ❓ Ask Your Question")

if mode == "Text":
    st.markdown('<div class="simple-card">', unsafe_allow_html=True)
    query_text = st.text_input(
        "**Type your question about herbs, cultivation, or regulations:**",
        placeholder="e.g., How to grow Ashwagandha in Maharashtra? What are the organic certification requirements?",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

elif mode == "Voice":
    st.markdown('<div class="simple-card">', unsafe_allow_html=True)
    st.markdown("#### 🎤 Speak Your Question")
    st.markdown("Click below and speak clearly in your selected language")
    
    # Voice recorder
    audio_data = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording", 
        key="recorder",
        use_container_width=True
    )
    
    if audio_data and "bytes" in audio_data:
        with st.spinner("🔄 Processing your voice... Please wait"):
            # Save mic recording temporarily (webm)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmpfile:
                tmpfile.write(audio_data["bytes"])
                tmpfile_path = tmpfile.name

            # Convert WebM → WAV safely using ffmpeg
            wav_path = tmpfile_path.replace(".webm", ".wav")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", tmpfile_path,
                    "-ar", "16000", "-ac", "1",
                    wav_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                st.error(f"❌ Audio conversion failed: {e.stderr.decode()}")
                wav_path = None

            # Recognize speech if conversion succeeded
            if wav_path:
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio = recognizer.record(source)
                try:
                    query_text = recognizer.recognize_google(audio, language=languages[user_lang])
                    st.success(f"**🗣️ You said:** {query_text}")
                except sr.UnknownValueError:
                    st.error("❌ Sorry, I could not understand the audio. Please try speaking more clearly.")
                except sr.RequestError as e:
                    st.error(f"❌ Could not request results; {e}")
    else:
        query_text = ""
        
    st.markdown('</div>', unsafe_allow_html=True)

# Load vector store (your existing code)
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

# Initialize Groq LLM (your existing code)
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Create RAG pipeline (your existing code)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# Function to clean text (your existing code)
def clean_text(text: str) -> str:
    text = re.sub(r"[*_#`]", "", text)
    text = text.replace("•", " ").replace("-", " ")
    text = text.replace(":", ": ")
    return text.strip()

# Get Answer Button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    get_answer_btn = st.button("🌱 Get Expert Answer", use_container_width=True)

# Process Query
if get_answer_btn:
    if not query_text or query_text.strip() == "":
        st.warning("⚠️ Please enter or speak a question first.")
    else:
        with st.spinner(f"🔍 Searching for the best answer in {user_lang}..."):
            try:
                # Display user's question
                st.markdown(f'<div class="simple-card"><h4>👤 Your Question:</h4><p>"{query_text}"</p></div>', unsafe_allow_html=True)
                
                # Translate query → English
                with st.spinner("🔄 Translating your question..."):
                    query_in_english = GoogleTranslator(source='auto', target='en').translate(query_text)

                # Get answer
                with st.spinner("🤔 Analyzing with expert knowledge..."):
                    answer = qa.run(query_in_english)

                # Translate back to user language
                with st.spinner("🔄 Translating answer to your language..."):
                    answer_translated = GoogleTranslator(source='en', target=languages[user_lang]).translate(answer)

                # Clean & show text
                answer_clean = clean_text(answer_translated)
                
                # Display answer in attractive box
                st.markdown(f"""
                <div class="answer-box">
                    <h3 style='color: #2E8B57; text-align: center;'>🌿 Expert Answer</h3>
                    <hr style='border: 1px solid #90EE90;'>
                    <div style='font-size: 1.1rem; line-height: 1.6; color: #2d5016;'>
                        {answer_clean}
                    </div>
                    <div style='text-align: center; margin-top: 15px;'>
                        <small style='color: #556B2F;'>💡 Powered by AI Agriculture Expert</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error processing your request: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌾 <strong>Kisaan Mitra</strong> - Empowering Farmers with AI Technology</p>
    <p><small>Always consult local agricultural experts for specific regional advice</small></p>
</div>
""", unsafe_allow_html=True)
