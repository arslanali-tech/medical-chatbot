from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import streamlit as st
import uuid
from datetime import datetime
import json
import sqlite3

import os
# def save_interaction(symptoms, response):
#     conn = sqlite3.connect('doctors-patient.db')
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO Interactions (timestamp, symptoms, response) VALUES (?, ?, ?)",
#                    (datetime.now(), symptoms, response))
#     conn.commit()
#     conn.close()
load_dotenv()
embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv("Openaiapikey"))
vectorstore = Chroma(persist_directory='chromadb', embedding_function=embedding_function, collection_name="doctors")
def store_interaction(symptoms, response):
    content=f'Symptoms:{symptoms}\nResponse:{response}'
    doc= Document(page_content=content)
    vectorstore.add_documents([doc])
    vectorstore.persist()
# === SESSION MANAGEMENT FUNCTIONS ===
def load_sessions():
    """Load all sessions from file"""
    try:
        with open("chat_sessions.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_sessions(sessions):
    """Save all sessions to file"""
    with open("chat_sessions.json", "w") as f:
        json.dump(sessions, f, indent=2)

def create_new_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return {
        "id": session_id,
        "title": f"Session {timestamp}",
        "messages": [],
        "created_at": timestamp
    }
def get_session_title(messages):
    """Generate a title based on first user message"""
    if messages:
        first_user_msg = next((msg[1] for msg in messages if msg[0] == "user"), "")
        if first_user_msg:
            # Take first 30 characters and add ellipsis if longer
            title = first_user_msg[:30] + ("..." if len(first_user_msg) > 30 else "")
            return title
    return "New Session"
retriever= vectorstore.as_retriever(search_kwargs={"k": 3})
def get_context(symptoms):
    vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.get_relevant_documents(symptoms)
    similar_cases=[]
    for case in results:
        text=case.page_content
        similar_cases.append(text)
    combined_text='/n---/n'.join(similar_cases)
    return combined_text
       
    return "\n".join(similar_cases)

with open("doctors.txt", "r") as file:
    doctors_list = file.read()

llm_openai=ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("Openaiapikey"),
)
template="""you are a medical assistant. Review the user's symptoms and examples from similar past cases.

### Similar Past Cases:
{context}

### Current User Input:
{symptoms}

If the input is incomplete, ask one clear follow-up question.

Only when you have enough info:
- Recommend an over-the-counter medicine.
- Recommend a doctor from the list below.

{doctors}

Make sure to give:
- Follow-up question (if needed), OR
- Medicine + Doctor Recommendation + Reason + Severity
"""

user_prompt=PromptTemplate(
    template=template,
    input_variables=["symptoms","doctors","context"],
)
llm_chain=LLMChain(
    llm=llm_openai,
    prompt=user_prompt
)
st.set_page_config(page_title="Medical Expert Chatbot", layout="centered")
def get_stored_data():
    try:
        results=vectorstore.get()
        documents=results['documents']
        if not documents:
            return []
        return documents
    except Exception as e:
        return [f'error loading history: {e}']

with st.sidebar:
    st.title("💬 Chat History")
    
    # Load all sessions
    all_sessions = load_sessions()
    
    # Button to create new session
    if st.button("➕ New Session", use_container_width=True):
        # Save current session if it has messages
        if "current_session_id" in st.session_state and "chat_history" in st.session_state:
            if st.session_state.chat_history:
                current_session = {
                    "id": st.session_state.current_session_id,
                    "title": get_session_title(st.session_state.chat_history),
                    "messages": st.session_state.chat_history,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                all_sessions[st.session_state.current_session_id] = current_session
                save_sessions(all_sessions)
        
        # Create new session
        new_session = create_new_session()
        st.session_state.current_session_id = new_session["id"]
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    st.subheader("Previous Sessions")
    sorted_sessions = sorted(all_sessions.items(), key=lambda x: x[1]["created_at"], reverse=True)
    for session_id, session in sorted_sessions:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(session["title"], key=session_id, use_container_width=True):
                st.session_state.current_session_id = session_id
                st.session_state.chat_history = session["messages"]
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{session_id}"):
                del all_sessions[session_id]
                save_sessions(all_sessions)
                if st.session_state.current_session_id == session_id:
                    new_session = create_new_session()
                    st.session_state.current_session_id = new_session["id"]
                    st.session_state.chat_history = []
                    st.rerun()
st.title("💡Intelligent Madical Assistant")
st.markdown("Ask your medical questions and get personalized recommendations.")
if "current_session_id" not in st.session_state:
    new_session = create_new_session()
    st.session_state.current_session_id = create_new_session()["id"]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if st.session_state.chat_history:
    current_title=get_session_title(st.session_state.chat_history)
    st.info(f"Current Session: {current_title}")
user_input = st.chat_input("Describe your symptoms:", key="user_input")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    context=get_context(user_input)
    bot_response = llm_chain.run(
            symptoms=user_input,
            doctors=doctors_list,
            context=context
        )
    st.session_state.chat_history.append(("bot", bot_response))
    store_interaction(user_input, bot_response)
    all_sessions = load_sessions()
    current_session={
            "id": st.session_state.current_session_id,
            "title": get_session_title(st.session_state.chat_history),
            "messages": st.session_state.chat_history,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    all_sessions[st.session_state.current_session_id] = current_session
    save_sessions(all_sessions)

for role, msg in st.session_state.chat_history:
        if role == 'user':
            st.chat_message('user').write(msg)
        else:
            st.chat_message('assistant').write(msg)    