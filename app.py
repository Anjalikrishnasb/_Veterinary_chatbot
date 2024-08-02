import streamlit as st
st.set_page_config(page_title="Veterinary Chatbot | Gemini", layout="wide")

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

# try:
#     import pyaudio
#     PYAUDIO_AVAILABLE = True
# except ImportError:
#     PYAUDIO_AVAILABLE = False
#     st.warning("PyAudio is not installed. Voice input functionality will be disabled.")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime
import random
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import base64
import re
import time
from PIL import Image
import imagehash
import fitz
import io
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import google.generativeai as genai
except ImportError:
    install('google-generativeai')
    import google.generativeai as genai

# Setup logging
logging.basicConfig(filename='chatbot.log', level=logging.DEBUG)

# Load environment variables
load_dotenv()
logging.debug("Environment variables loaded")

def get_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except KeyError:
            st.error("GOOGLE_API_KEY not found in environment or Streamlit secrets.")
            st.stop()
    return api_key
    
api_key = get_api_key()
genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel('models/gemini-1.5-pro')
    print("Gemini model successfully loaded")
except Exception as e:
    st.error(f"Failed to load Gemini model: {str(e)}")
    st.stop()

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'current_image_match' not in st.session_state:
    st.session_state['current_image_match'] = None
if 'current_image_data' not in st.session_state:
    st.session_state['current_image_data'] = None
if 'current_image_content' not in st.session_state:
    st.session_state['current_image_content'] = None
if 'voice_input' not in st.session_state:
    st.session_state.voice_input = ""

def get_pdf_text_from_folder(pdf_folder):
    text = ""
    try:
        for pdf_file in os.listdir(pdf_folder):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, pdf_file)
                loader = PyPDFLoader(pdf_path)
                pages = loader.load_and_split()
                for page in pages:
                    text += page.page_content + "\n\n"
    except Exception as e:
        logging.error(f"Error reading PDFs from folder {pdf_folder}: {e}")
        st.error(f"Error reading PDFs. Please check the log for details.")
    return text

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000,length_function=len)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        st.error(f"Error splitting text: {e}")
        return []
    
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error(f"Error processing text. Please check the log for details.")
        return None
    
def get_conversational_chain():
    prompt_template = """
    You are a knowledgeable veterinary assistant. Use the information provided in the context, chat history, and image context to answer the question accurately and concisely. If the answer cannot be found in the provided information, state that clearly.

    Context:
    {context}

    Chat History:
    {chat_history}

    Image Context:
    {image_context}

    Human: {question}
    AI: Based on the provided information:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history","image_context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logging.error(f"Error loading QA chain: {e}")
        st.error(f"Error setting up the conversation chain. Please check the log for details.")
        return None
    
def extract_text_from_pdf(pdf_path, page_number):
    try:
        with fitz.open(pdf_path) as doc:
            if page_number < len(doc):
                page = doc[page_number]
                text = page.get_text()
            else:
                text = "Page number out of range"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""
    
def process_image(uploaded_file):
    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file)
            uploaded_hash = imagehash.average_hash(uploaded_image)
            
            data_folder = os.path.join(os.path.dirname(__file__), "data")
            st.write(f"Debug: Searching in folder {data_folder}")
            
            if not os.path.exists(data_folder):
                st.error(f"Error: Folder {data_folder} does not exist")
                return None, None, None

            for filename in os.listdir(data_folder):
                if filename.lower().endswith('.pdf'):
                    pdf_path = os.path.join(data_folder, filename)
                    st.write(f"Debug: Processing PDF {filename}")
                    
                    with fitz.open(pdf_path) as doc:
                        for idx in range(len(doc)):  
                            page = doc[idx]
                            images = page.get_images(full=True)
                            for image_index, img in enumerate(images):
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                image = Image.open(io.BytesIO(image_bytes))
                                
                                image_hash = imagehash.average_hash(image)
                                hash_diff = uploaded_hash - image_hash
                                
                                if hash_diff < 15:  
                                    st.write(f"Debug: Match found in {filename}, image {idx}")
                                    pdf_text = extract_text_from_pdf(pdf_path, idx)
                                    return f"{os.path.splitext(filename)[0]}_image_{idx}", image, pdf_text
            st.write("Debug: No matching image found")
            return None, None, None
        except Exception as e:
            st.error(f"Error processing uploaded image: {str(e)}")
            return None, None, None
    return None, None, None

def user_input(user_question, chat_history, image_match=None, image_content=None):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        recent_context = "\n".join([f"Human: {chat['question']}\nAI: {chat['answer']}" for chat in chat_history[-5:]])
        combined_query = f"{recent_context}\nHuman: {user_question}"
        
        docs = new_db.similarity_search(combined_query,k=10)
        context = "\n".join([doc.page_content for doc in docs])

        if image_match and image_content:
            context += f"\nImage context: {image_match}\n{image_content}"

        chain = get_conversational_chain()
        
        response = chain(
            {
                "input_documents": docs,
                "question": combined_query,  
                "chat_history": recent_context,
                "context": context,
                "image_context": image_content if image_content else "No image context available"
            },
            return_only_outputs=True
        )
        
        if not response or not response.get("output_text"):
            return "I'm sorry, I don't have that information in my existing data."
        else:
            return format_response(response["output_text"])
            
    except Exception as e:
        logging.error(f"Error in user_input function: {e}")
        return f"Sorry, something went wrong. Please try again later. Error: {str(e)}"
    
def format_response(response):
    response = response.replace(' - ', ': ').replace('•', '*')
    response = re.sub(r'(\d+)', r'\n\1.', response)  
    response = re.sub(r'\n\s*\n', '\n', response) 
    return response.strip()

def get_greeting():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good morning! How can I help you today?"
    elif 12 <= current_hour < 18:
        return "Good afternoon! How can I assist you today?"
    else:
        return "Good evening! What can I do for you today?"

def text_to_speech(text):
    try:
        tts = gTTS(text)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_base64 = base64.b64encode(audio_fp.read()).decode()
        return audio_base64
    except Exception as e:
        logging.error(f"Error in text_to_speech function: {e}")
        st.error(f"Error converting text to speech. Please check the log for details.")
        return None

def play_audio(audio_fp):
    try:
        audio_fp.seek(0)
        audio = AudioSegment.from_file(audio_fp, format="mp3")
        play(audio)
    except Exception as e:
        logging.error(f"Error playing audio: {e}")
        st.error(f"Error playing audio. Please check the log for details.")

def speech_to_text():
    try:
        import speech_recognition as sr
    except ImportError:
        st.error("Speech recognition is not available. PyAudio or speech_recognition module is missing.")
        return None
    
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            warning_placeholder = st.empty()  
            warning_placeholder.warning("Listening... (Will stop after 3 seconds of silence)")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
               audio = recognizer.listen(source, timeout=3)
               text = recognizer.recognize_google(audio).lower()
               warning_placeholder.empty()
               return text
            except sr.UnknownValueError:
                st.error("Sorry, I couldn't understand that.")
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
    except AttributeError:
        st.error("Microphone is not available. Please make sure PyAudio is installed and you have a working microphone.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    return None

faq = {
    "What are the vaccination schedules for dogs?": "Vaccinations 💉 typically start at 6-8 weeks of age and continue every 3-4 weeks until 16 weeks old.",
    "How often should I take my pet to the vet?": "It's recommended to take your pet to the vet at least once a year for a check-up 🏥.",
    "What should I do if my pet is vomiting?": "If your pet is vomiting, withhold food for 12-24 hours and offer small amounts of water. If it continues, consult a veterinarian 👩🏻‍⚕️.",
    "What are the signs of a healthy pet?": "Signs of a healthy pet 🐕 include a shiny coat, clear eyes, good appetite, and regular bowel movements.",
    "How can I tell if my pet is in pain?": "Signs of pain 😿 in pets include limping, decreased activity, vocalizing, and changes in eating or grooming habits.",
    "What should I do if my pet has diarrhea?": "If your pet has diarrhea, ensure they stay hydrated and withhold food for 12-24 hours. If it persists, consult a veterinarian 👩🏻‍⚕️.",
    "What are common signs of allergies in pets?": "Common signs of allergies in pets 🐏 include itching, licking, ear infections, and gastrointestinal issues.",
    "How can I prevent ticks and fleas on my pet?": "Use preventative medications, regularly check your pet for ticks and fleas, and maintain a clean environment 🛖.",
    "What should I feed my pet?": "Provide a balanced diet with appropriate portions of high-quality pet food, and avoid feeding them human food 🦴.",
    "How can I maintain my pet's dental health?": "Regularly brush your pet's teeth 🦷, provide dental chews, and schedule professional cleanings with your veterinarian.",
    "What is the best way to train my dog?": "Consistency, positive reinforcement, and patience are key. Reward good behavior with treats and praise.",
    "How often should I bathe my pet?": "Generally, dogs should be bathed every 4-6 weeks, while cats usually groom themselves and need fewer baths.",
    "What should I do if my pet is overweight?":"Consult your vet for a weight management plan, which may include a special diet and increased exercise.",
    "How do I know if my pet is dehydrated?":"Signs of dehydration include dry gums, lethargy, and loss of skin elasticity. Ensure they always have access to fresh water.",
    "What are the signs of heatstroke in pets?":"Symptoms include excessive panting, drooling, vomiting, and weakness. Move your pet to a cool area and contact a vet immediately.",
    "How can I help my pet with separation anxiety?":"Gradual desensitization, creating a safe space, and using calming aids can help. Consult your vet for additional advice.",
    "Why is my pet scratching excessively?":"Excessive scratching can be due to allergies, parasites, or skin conditions. A vet visit is recommended for proper diagnosis.",
    "How can I keep my pet's coat shiny and healthy?":"Regular grooming, a balanced diet rich in omega-3 fatty acids, and proper hydration help maintain a healthy coat.",
    "What should I do if my pet eats something toxic?":"Contact your vet or an emergency animal clinic immediately. Keep the packaging or a sample of the substance if possible.",
    "How can I prevent my pet from getting lost?":"Use ID tags, microchips, and ensure your pet is always supervised when outdoors.",
    "What is the best way to introduce a new pet to my home?":"Gradually introduce them to each other, use positive reinforcement, and give them time to adjust.",
    "How do I know if my pet is getting enough exercise?":"Monitor their weight, behavior, and overall health. Active and engaged pets usually get enough exercise.",
    "What are common signs of arthritis in pets?":"Symptoms include limping, difficulty rising, reluctance to jump or climb stairs, and stiffness after resting.",
    "How can I help my pet during thunderstorms or fireworks?":"Create a safe space, use calming products, and consider desensitization training. Consult your vet for more options.",
    "Why is my pet eating grass?":"Pets may eat grass due to boredom, dietary deficiency, or to induce vomiting. Monitor their behavior and consult a vet if it persists.",
    "What should I do if my pet is constipated?":"Ensure they have plenty of water and fiber in their diet. If constipation persists, consult your vet.",
    "How can I prevent my pet from chewing on household items?":"Provide plenty of toys and chews, ensure they get enough exercise, and use positive reinforcement for good behavior.",
    "What are the signs of diabetes in pets?":"Increased thirst, frequent urination, weight loss, and lethargy are common signs. A vet visit is necessary for diagnosis and treatment.",
    "How do I clean my pet's ears?":"Use a vet-approved ear cleaner and gently wipe the outer ear with a cotton ball. Avoid inserting anything into the ear canal.",
    "What should I do if my pet has a seizure?":"Stay calm, keep them safe from injury, and time the seizure. Contact your vet immediately after the seizure ends."
}

health_tips = [
    "Make sure your pet gets regular check-ups.",
    "Keep your pet's vaccinations up to date.",
    "Provide a balanced diet for your pet.",
    "Ensure your pet gets enough exercise.",
    "Maintain proper grooming for your pet.",
    "Provide fresh, clean water at all times to keep your pet hydrated.",
    "Use flea, tick, and worm preventatives regularly.",
    "Brush your pet’s teeth regularly and provide dental chews to prevent dental disease.",
    "Feed your pet a balanced diet with high-quality ingredients appropriate for their age and size.",
    "Keep an eye on your pet’s weight and adjust their diet and exercise accordingly.",
    "Provide a comfortable, clean, and safe sleeping area for your pet.",
    "Provide toys and activities to keep your pet mentally stimulated and prevent boredom.",
    "Socialize your pet with other animals and people to reduce anxiety and improve behavior.",
    "Clean your pet’s ears regularly to prevent infections.",
    "Watch for signs of illness such as changes in appetite, behavior, or energy levels, and consult a vet if needed.",
    "Ensure your home and yard are safe and free from hazards like toxic plants and chemicals.",
    "Protect your pet from extreme temperatures; provide warmth in winter and cool areas in summer.",
    "Consider spaying or neutering your pet to prevent unwanted litters and reduce certain health risks.",
    "Perform regular at-home check-ups, including checking their eyes, ears, and skin for any abnormalities.",
    "Use a harness or a crate when traveling to keep your pet safe.",
    "Have a pet first aid kit and a plan in place for emergencies."
]

def main():
     
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'current_image_match' not in st.session_state:
        st.session_state['current_image_match'] = None
    if 'current_image_data' not in st.session_state:
        st.session_state['current_image_data'] = None
    if 'current_image_content' not in st.session_state:
        st.session_state['current_image_content'] = None
    if 'voice_input' not in st.session_state:
        st.session_state.voice_input = ""
    if 'last_processed_question' not in st.session_state:
        st.session_state.last_processed_question = ""

    def load_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
            return encoded_image
        except FileNotFoundError:
            st.error(f"Image file not found:{image_path}")
            return None

    image_path = os.path.join("images", "pet-friendly-chalk-white-icon-on-black-background-vector.jpg")
    encoded_image = load_image(image_path)
    if encoded_image is None:
        st.error("Failed to load the image. Using a placeholder or default image.")
    
    # Custom CSS for improved styling
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .fixed-top {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: white;
        z-index: 1000;
        padding: 1rem;
    }
    .header {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }
    .header img {
        width: 60px;
        margin-right: 1rem;
    }
    .header h1 {
        color: #50C878;
        font-size:2rem;
    }
           
    .health-tip {
        background-color: #50C878;
        padding: 0.2rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .health-tip h3 {
        color: white; /* Different pink shade for the heading text */
    }
    
    .response-header {
    color: #50C878;
    font-size: 1.5rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    }  
    .chat-container {
        display: flex;
        align-items: flex-end;
        gap: 2rem;
                
    }
    .chat-main {
        flex: 2;
    }
    .chat-sidebar {
        flex: 1;
    }
    .input-container {
        display: flex;
        align-items: flex-end;
        margin-top: 1rem;
    }
    .input-container input {
        flex-grow: 1;
        margin-right: 0.5rem;
    }
    .stTextInput > div > div > input {
        height:45px;
    }
    .stButton > button {
                
        height: 45px;
        width: 100%;
        margin: 0;
    }
    .button-container {
        display: flex;
        gap: 10px;
    }
    .button-label {
    font-size: 12px;
    margin-bottom: 2px;
    text-align: center;
}
    .custom-column {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }       
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(
        f"""
        
        <div class="header">
        <img src="data:image/jpeg;base64,{encoded_image}" alt="Veterinary Icon"/>
        <h1>PAWSITIVE</h1>
    </div>
    """, unsafe_allow_html=True)

    # Health Tip
    health_tip = random.choice(health_tips)
    st.markdown(f"""
    <div class="health-tip">
        <h3>Health Tip of the Day📋</h3>
        <p>{health_tip}</p>
    </div>
    """, unsafe_allow_html=True)

    # Main chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat main area
    st.markdown('<div class="chat-main">', unsafe_allow_html=True)
    
    if 'show_image' not in st.session_state:
        st.session_state.show_image = True
        
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if st.session_state.show_image:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Processing image..."):
            image_match, matched_image,image_content = process_image(uploaded_file)
        if image_match and matched_image and image_content:
            st.warning("This image is similar to an image in my document. The match may not be 100% accurate.", icon="⚠️")
            time.sleep(3)
            st.empty()
            st.write(f"Image matched: {image_match}")
            st.session_state.current_image_match = image_match
            st.session_state.current_image_data = matched_image
            st.session_state.current_image_content = image_content
            
        else:
            st.warning("No matching image found in the existing data.")
            st.session_state.current_image_match = None
            st.session_state.current_image_data = None
            st.session_state.current_image_content = None

    if st.session_state.current_image_data is not None:
        st.image(st.session_state.current_image_data, caption="Current Image Context", use_column_width=True)

    # User input
    col1, col2, col3 = st.columns([6,1,1])

    if st.session_state.voice_input:
        user_question = st.session_state.voice_input
        st.session_state.voice_input = ""
    else:
        user_question = st.session_state.get('user_input', '')

    with col1:
        user_question = st.text_input("Ask me anything related to pet care or the uploaded image!", key="user_input", value=user_question,  placeholder=get_greeting())
    
    with col2:
        st.markdown("<p class='button-label'>PRESS</p>", unsafe_allow_html=True)
        speak_button = st.button("🎤")
    with col3: 
        st.markdown("<p class='button-label'>PRESS</p>", unsafe_allow_html=True)
        send_button = st.button("➤")

    if speak_button:
        try:
            recognized_text = speech_to_text()
            if recognized_text:
                user_question = recognized_text
                st.session_state.voice_input = recognized_text
                st.experimental_rerun()
        except Exception as e:
            st.error(f"An error occurred with speech recognition: {str(e)}")
            st.info("Speech recognition may not be available in this environment. Please type your question instead.")

    

    if send_button or (user_question and user_question != st.session_state.last_processed_question):
            st.markdown("<h3>Response:</h3>", unsafe_allow_html=True)
            response_placeholder = st.empty()
            full_response = user_input(user_question, 
                                       st.session_state.chat_history, 
                                       st.session_state.current_image_match,
                                       st.session_state.current_image_content)
            displayed_response = ""
            for word in full_response.split():
                displayed_response += f" {word}"
                response_placeholder.markdown(displayed_response)
                time.sleep(0.02)
            
            audio_base64 = text_to_speech(full_response)
            
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": full_response.strip(),
                "audio": audio_base64
            })

            st.session_state.last_processed_question = user_question
            
    for chat in st.session_state.chat_history:
        with  st.expander(f"**🐰**: {chat['question']}"):
            st.markdown(f"**🤖**: {chat['answer']}")
            if chat['audio']:
                st.markdown(f'<audio src="data:audio/mp3;base64,{chat["audio"]}" controls></audio>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar
    st.markdown('<div class="chat-sidebar">', unsafe_allow_html=True)
    
    st.sidebar.title("FAQ")
    question = st.sidebar.selectbox("Select a question:", list(faq.keys()))
    if question:
        st.sidebar.write(f"**Answer**: {faq[question]}")

    st.sidebar.title("Chat History")
    for chat in st.session_state.chat_history:
        with st.sidebar.expander(f"**🐰**: {chat['question']}"):
            st.sidebar.markdown(f"**🤖**: {chat['answer']}")
            if chat['audio']:
                st.sidebar.markdown(f'<audio src="data:audio/mp3;base64,{chat["audio"]}" controls></audio>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()