import streamlit as st
from medical_bot import MedicalBot
import time
import os
import random

# Set page config
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Dark theme colors */
    .main {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: #0f172a;
    }
    .stMarkdown, p, h1, h2, h3, h4, h5, h6, li {
        color: #e2e8f0 !important;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    .chat-message:hover {
        transform: translateY(-2px);
    }
    .chat-message.user {
        background-color: #1e293b;
        border-left: 5px solid #3b82f6;
        color: #ffffff;
    }
    .chat-message.assistant {
        background-color: #1e3a2f;
        border-left: 5px solid #10b981;
        color: #ffffff;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .content {
        margin-top: 0.5rem;
        line-height: 1.6;
        font-size: 1.1rem;
    }
    .stTextInput>div>div>input {
        border-radius: 1rem;
        padding: 12px 20px;
        background-color: #1e293b;
        color: #ffffff;
        border: 2px solid #334155;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    .stButton>button {
        border-radius: 1rem;
        padding: 12px 24px;
        background-color: #3b82f6;
        color: white;
        border: none;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
    }
    .typing-indicator {
        display: flex;
        align-items: center;
        margin-top: 0.5rem;
    }
    .typing-indicator span {
        height: 8px;
        width: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
        animation: typing 1s infinite;
    }
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    @keyframes typing {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    .welcome-message {
        text-align: center;
        padding: 2.5rem;
        background: linear-gradient(135deg, #1e3a2f 0%, #1e293b 100%);
        border-radius: 1.5rem;
        margin-bottom: 2rem;
        animation: fadeIn 1s;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Highlight important information */
    .highlight {
        background-color: #334155;
        padding: 0.3rem 0.6rem;
        border-radius: 0.5rem;
        font-weight: bold;
        color: #ffffff;
    }
    /* Make links more visible */
    a {
        color: #3b82f6 !important;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    a:hover {
        color: #60a5fa !important;
        text-decoration: underline;
    }
    /* Improve readability of code blocks */
    code {
        background-color: #334155;
        color: #e2e8f0;
        padding: 0.3rem 0.6rem;
        border-radius: 0.5rem;
        font-family: 'Fira Code', monospace;
    }
    /* History sidebar styling */
    .history-item {
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-radius: 1rem;
        background-color: #1e293b;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .history-item:hover {
        background-color: #334155;
        transform: translateX(5px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .history-timestamp {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .history-question {
        font-weight: bold;
        margin: 0.4rem 0;
        color: #e2e8f0;
    }
    .history-source {
        font-size: 0.9rem;
        color: #10b981;
    }
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%);
    }
    /* Delete button styling */
    .delete-btn {
        background-color: #ef4444;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 6px 12px;
        font-size: 0.9rem;
        cursor: pointer;
        margin-top: 0.5rem;
        transition: all 0.3s ease;
    }
    .delete-btn:hover {
        background-color: #dc2626;
        transform: translateY(-1px);
    }
    /* Clear history button styling */
    .clear-history-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: none;
        border-radius: 1rem;
        padding: 10px 20px;
        font-weight: bold;
        margin-bottom: 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .clear-history-btn:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        transform: translateY(-1px);
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0f172a;
    }
    .css-1d391kg .sidebar-content {
        background-color: #1e293b;
        border-radius: 1rem;
        padding: 1rem;
        margin: 1rem;
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #3b82f6;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #2563eb;
    }
    /* Auto-scrolling styles */
    .stTextInput {
        position: fixed;
        bottom: 3rem;
        background-color: #1e293b;
        z-index: 100;
    }
    .main .block-container {
        padding-bottom: 5rem;
    }
    .stMarkdown {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history and bot
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi! I'm your personal medical AI assistant. How can I help you today?"})

if "bot" not in st.session_state:
    st.session_state.bot = None

if "processing" not in st.session_state:
    st.session_state.processing = False

if "delete_index" not in st.session_state:
    st.session_state.delete_index = None

if "clear_history" not in st.session_state:
    st.session_state.clear_history = False

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}

if "searching_pdf" not in st.session_state:
    st.session_state.searching_pdf = False

if "searching_wiki" not in st.session_state:
    st.session_state.searching_wiki = False

# Main app
def main():
    st.title("🏥 Medical AI Assistant")
    
    # Initialize bot with loading indicator
    if st.session_state.bot is None:
        with st.spinner("Initializing medical knowledge base... This may take a minute..."):
            pdf_paths = [
                "Gale Encyclopedia of Medicine. Vol. 2. 2nd ed.pdf",
                "Gale Encyclopedia of Medicine. Vol. 1. 2nd ed (1).pdf"
            ]
            st.session_state.bot = MedicalBot(pdf_paths)
    
    # Sidebar for query history
    with st.sidebar:
        st.header("Query History")
        
        # Clear history button
        if st.button("🗑️ Clear All History", key="clear_history_btn"):
            st.session_state.clear_history = True
        
        # Handle clear history
        if st.session_state.clear_history:
            if st.session_state.bot.clear_history():
                st.success("History cleared successfully!")
                st.session_state.clear_history = False
                st.rerun()
            else:
                st.error("Failed to clear history.")
                st.session_state.clear_history = False
        
        # Get recent history from bot
        recent_history = st.session_state.bot.get_recent_history(limit=10)
        
        if recent_history:
            for i, item in enumerate(reversed(recent_history)):
                with st.expander(f"{item['timestamp']} - {item['question'][:50]}..."):
                    st.markdown(f"""
                    <div class="history-item">
                        <div class="history-timestamp">{item['timestamp']}</div>
                        <div class="history-question">{item['question']}</div>
                        <div class="history-source">Source: {item['source']}</div>
                        <div class="history-answer">{item['answer']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Delete button for each history item
                    if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                        actual_index = len(recent_history) - 1 - i
                        if st.session_state.bot.delete_query(actual_index):
                            st.success("Query deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete query.")
        else:
            st.info("No query history yet. Start asking questions to build your history!")
    
    # Display welcome message
    if len(st.session_state.messages) == 1:
        st.markdown("""
        <div class="welcome-message">
            <h2>Welcome to Your Medical AI Assistant</h2>
            <p>Ask me any medical questions, and I'll provide clear, detailed answers.</p>
            <p>I'll remember our conversation and can reference previous questions you've asked.</p>
            <p>You can ask follow-up questions using words like "same", "it", "this", or "that".</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages with animations
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask your medical question..."):
        # Check if we have a cached response for this exact question
        if prompt in st.session_state.response_cache:
            response = st.session_state.response_cache[prompt]
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
            return
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"""
            <div class="chat-message user">
                <div class="content">{prompt}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Get bot response with typing animation
        with st.chat_message("assistant"):
            # Create a placeholder for the response
            response_placeholder = st.empty()
            
            # Show typing indicator
            response_placeholder.markdown("""
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Create status indicators
            status_container = st.empty()
            
            # Simulate progress while waiting for response
            for i in range(100):
                time.sleep(0.02)  # Reduced from 0.03 for faster progress
                progress_bar.progress(i + 1)
                
                # Update status messages
                if i < 30:
                    status_container.info("Searching medical knowledge base...")
                elif i < 60:
                    status_container.info("Analyzing relevant information...")
                else:
                    status_container.info("Generating response...")
                
                if i % 20 == 0:  # Update typing indicator every 20%
                    response_placeholder.markdown("""
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Get the actual response with context
            context = st.session_state.last_question if st.session_state.last_question else None
            # Check if this is a follow-up question
            is_follow_up = False
            if context and any(word in prompt.lower() for word in ["same", "it", "this", "that", "these", "those", "they", "them", "their", "its"]):
                is_follow_up = True
            
            # Get response from bot
            response = st.session_state.bot.query(prompt, context)
            
            # Cache the response for future use
            st.session_state.response_cache[prompt] = response
            
            # Update last question and answer for context
            st.session_state.last_question = prompt
            st.session_state.last_answer = response
            
            # Clear progress bar and typing indicator
            progress_bar.empty()
            status_container.empty()
            response_placeholder.empty()
            
            # Display response with animation
            response_placeholder.markdown(f"""
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.5)
            response_placeholder.markdown(f"""
            <div class="chat-message assistant">
                <div class="content">{response}</div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

