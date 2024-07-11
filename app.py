
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from datetime import datetime
import random
import pandas as pd


from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text_from_folder(pdf_folder):
    text=""
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If 
    the answer is not in the provided context, just say, "Apologies, I am unable to find the answer." Don't provide the wrong answer.

Context:
{context}?

Question:
{question}

Answer:
"""

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt= PromptTemplate(template=prompt_template,input_variables={"context","question"})

    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    
    return chain

def user_input(user_question):
    greetings = ["hello", "hai", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if user_question.lower() in greetings:
        return get_greeting()

    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    chain=get_conversational_chain()
    response=chain(
        {"input_documents":docs,"question":user_question}
        ,return_only_outputs=True
    )
    print(response)
    st.write("Reply:",response["output_text"])
 

def get_greeting():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good morning! How can I help you today?"
    elif 12 <= current_hour < 18:
        return "Good afternoon! How can I assist you today?"
    else:
        return "Good evening! What can I do for you today?"

faq = {
    "What are the vaccination schedules for dogs?": "Vaccinations 💉 typically start at 6-8 weeks of age and continue every 3-4 weeks until 16 weeks old.",
    "How often should I take my pet to the vet?": "It's recommended to take your pet to the vet at least once a year for a check-up 🏥.",
    "What should I do if my pet is vomiting?": "If your pet is vomiting, withhold food for 12-24 hours and offer small amounts of water. If it continues, consult a veterinarian 👩🏻‍⚕️.",
    "What are the signs of a healthy pet?": "Signs of a healthy pet 🐕 include a shiny coat, clear eyes, good appetite, and regular bowel movements.",
    "How can I tell if my pet is in pain?": "Signs of pain 😿 in pets include limping, decreased activity, vocalizing, and changes in eating or grooming habits.",
    "What should I do if my pet has diarrhea?":"If your pet has diarrhea, ensure they stay hydrated and withhold food for 12-24 hours. If it persists, consult a veterinarian 👩🏻‍⚕️.",
    "What are common signs of allergies in pets?":"Common signs of allergies in pets 🐏 include itching, licking, ear infections, and gastrointestinal issues.",
    "How can I prevent ticks and fleas on my pet?":"Use preventative medications, regularly check your pet for ticks and fleas, and maintain a clean environment 🛖.",
    "What should I feed my pet?":"Provide a balanced diet with appropriate portions of high-quality pet food, and avoid feeding them human food 🦴.",
    "ow can I maintain my pet's dental health?":"Regularly brush your pet's teeth 🦷, provide dental chews, and schedule professional cleanings with your veterinarian."

}

health_tips = [
    "Ensure your pet has fresh water available at all times🫗",
    "Regular exercise is important for your pet's health 🐕‍🦺",
    "Brush your pet's teeth regularly to prevent dental issues 🪥",
    "Keep your pet's vaccinations up to date 💉",
    "Provide a balanced diet for your pet's nutritional needs 🍏",
    "Maintain proper grooming to keep the coat clean and prevent skin infections 🐈‍⬛",
    "Offer toys, training, and social interaction to prevent behavioral problems 🧸"
    ""

]

def main():
    st.set_page_config("Chat |PDF")
    st.header("Veterinary chatbot🐶🩺")

    st.write(get_greeting())
    st.sidebar.title("Veterinary chatbot")
    st.sidebar.write("**Frequently Asked Questions**")
    question = st.sidebar.selectbox("Select a question:", list(faq.keys()))
    if question:
        st.sidebar.write(f"**Answer**: {faq[question]}")

    
    for chat in st.session_state.get("chat_history", []):
        st.sidebar.write(f"**User**: {chat['question']}")
        st.sidebar.write(f"**Bot**: {chat['answer']}")

        
    folder_path = "C:\\Users\\ANJALI\\OneDrive\\Desktop\\Gemini\\data"
    with st.spinner("Processing..."):
        raw_text = get_pdf_text_from_folder(folder_path)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    st.write("**Daily Pet Health Tip:**",random.choice(health_tips))

    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    
    st.markdown(
        """
        <style>
        
        .input-container input {
            border: none;
            outline: none;
            width: 100%;
            padding: 8px;
        }
        .input-container button {
            border: none;
            background-color: #4CAF50;
            color: white;
            padding: 8px;
            border-radius: 5px;
            cursor: pointer;
        }
        .input-container button:hover {
            background-color: #45a049;
        }
        </style>
        """, unsafe_allow_html=True
    )

    with st.form(key='my_form', clear_on_submit=True):
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        user_question = st.text_input("", key="input_box", placeholder="Ask a question")
        submit_button = st.form_submit_button(label='Submit')
        st.markdown('</div>', unsafe_allow_html=True)
        if submit_button and user_question:
            response = user_input(user_question)
            st.write("Reply:", response)       
        
            if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []
                    st.session_state["chat_history"].append({"question": user_question, "answer": st.session_state.get("response", "")})

    st.sidebar.write("**Chat History**")
    if "chat_history" in st.session_state:
        for chat in st.session_state["chat_history"]:
            st.sidebar.write(f"**User**: {chat['question']}")
            st.sidebar.write(f"**Bot**: {chat['answer']}")

if __name__=="__main__":
    main()