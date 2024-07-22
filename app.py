import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import StreamingStdOutCallbackHandler
import os
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime
import random
import logging
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import base64
import re
import time

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
logging.basicConfig(filename='chatbot.log', level=logging.DEBUG)

def get_pdf_text_from_folder(pdf_folder):
    text = ""
    try:
        for pdf_file in os.listdir(pdf_folder):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, pdf_file)
                pdf_reader = PdfReader(pdf_path)
                for page in pdf_reader.pages:
                    text += page.extract_text()
    except Exception as e:
        logging.error(f"Error reading PDFs from folder {pdf_folder}: {e}")
        st.error(f"Error reading PDFs. Please check the log for details.")
    return text

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
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
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error(f"Error processing text. Please check the log for details.")
        return []

def get_conversational_chain():
    prompt_template = """
You are a knowledgeable veterinary assistant. Use the provided context to answer the question accurately and concisely. If the answer is not in the context, respond with, "I'm sorry, I don't have that information."

Context:
{context}

Question:
{question}

Answer:
"""
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logging.error(f"Error loading QA chain: {e}")
        st.error(f"Error setting up the conversation chain. Please check the log for details.")
        return None

def user_input(user_question):
    try:
        greetings = ["hello", "hai", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if user_question.lower() in greetings:
            yield from get_greeting().split()
            return

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        streaming_handler = StreamingStdOutCallbackHandler()

        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            callbacks=[streaming_handler],
            return_only_outputs=True
        )
        if not response or not response.get("output_text"):
            yield from "Apologies, I am unable to find the answer. Can you please rephrase your question?".split()

        else:
            formatted_response = format_response(response["output_text"])
            for word in formatted_response.split():
                yield word
    except Exception as e:
        logging.error(f"Error in user_input function: {e}")
        yield from f"Sorry, something went wrong. Please try again later. Error: {str(e)}".split()

def format_response(response):
    response = response.replace(' - ', ': ').replace('•', '*')
    response = re.sub(r'(\d+)', r'\n\1.', response)  # Adding new lines before numbered points
    response = re.sub(r'\n\s*\n', '\n', response)  # Removing multiple new lines
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
        return audio_fp
    except Exception as e:
        logging.error(f"Error in text_to_speech function: {e}")
        st.error(f"Error converting text to speech. Please check the log for details.")
        return None

def play_audio(audio_fp):
    try:
        audio = AudioSegment.from_file(audio_fp, format="mp3")
        play(audio)
    except Exception as e:
        logging.error(f"Error playing audio: {e}")
        st.error(f"Error playing audio. Please check the log for details.")

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        warning_placeholder = st.empty()  # Create a placeholder for the warning
        warning_placeholder.warning("Listening... (Will stop after 3 seconds of silence)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
           audio = recognizer.listen(source, timeout=3)
           text = recognizer.recognize_google(audio).lower()
           warning_placeholder.empty()
           return text
        except sr.UnknownValueError:
           warning_placeholder.empty()
           st.error("Sorry, I couldn't understand that.")
           return None
        except sr.RequestError as e:
           warning_placeholder.empty()
           st.error(f"Could not request results; {e}")
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
    st.set_page_config(page_title="Veterinary Chatbot | Gemini", layout="wide")
    def load_image(image_path):
        with open(image_path, "rb") as image_file:
             encoded_image = base64.b64encode(image_file.read()).decode()
        return encoded_image
    
    image_path = r"C:\Users\ANJALI\OneDrive\Desktop\_Veterinary_chatbot\pet-friendly-chalk-white-icon-on-black-background-vector.jpg"
    encoded_image = load_image(image_path)
    
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
    
    
    # User input
    
    col1, col2,col3 = st.columns([6,1,1])
    with col1:
        user_question = st.text_input("Hi,Ask me anything related to pet care!", key="user_input",placeholder=get_greeting())
       
    with col2:
        st.markdown("<p class='button-label'>PRESS</p>", unsafe_allow_html=True)
        speak_button = st.button("🎤")
    with col3: 
        st.markdown("<p class='button-label'>PRESS</p>", unsafe_allow_html=True)
        send_button = st.button("➤")
        
    if speak_button:
        recognized_text = speech_to_text()  # Call the speech recognition function
        if recognized_text:
            user_question = recognized_text
            st.session_state.user_input_text = user_question  # Update the user question with recognized text
            st.text_input("", value=user_question, key="recognized_input", placeholder=get_greeting())
            if user_question:
                st.markdown("<h3>Response:</h3>", unsafe_allow_html=True)
                response_placeholder = st.empty()
                full_response = ""
                for word in user_input(user_question):
                    full_response += word + " "
                    response_placeholder.markdown(full_response)
                    time.sleep(0.05)
                
                audio_fp = text_to_speech(full_response)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3')

                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({"question": user_question, "answer": full_response.strip()})
                st.session_state.user_question = ""
    if send_button:
        if user_question:
            st.markdown("<h3>Response:</h3>", unsafe_allow_html=True)
            response_placeholder = st.empty()
            full_response = ""
            for word in user_input(user_question):
                full_response += word + " "
                response_placeholder.markdown(full_response)
                time.sleep(0.05)
            
            audio_fp = text_to_speech(full_response)
            if audio_fp:
                st.audio(audio_fp, format='audio/mp3')

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append({"question": user_question, "answer": full_response.strip()})
            st.session_state.user_question = ""

    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar
    st.markdown('<div class="chat-sidebar">', unsafe_allow_html=True)
    
    st.sidebar.title("FAQ")
    question = st.sidebar.selectbox("Select a question:", list(faq.keys()))
    if question:
        st.sidebar.write(f"**Answer**: {faq[question]}")

    st.sidebar.title("Chat History")
    if "chat_history" in st.session_state:
        for chat in st.session_state.chat_history:
            st.sidebar.markdown(f"**🐰**: {chat['question']}")
            st.sidebar.markdown(f"**🤖**: {chat['answer']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()