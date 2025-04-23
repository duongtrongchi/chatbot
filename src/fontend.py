import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Medical chatbot system prompt
MEDICAL_SYSTEM_PROMPT = """
You are MediAssist, a helpful medical assistant chatbot designed to provide general medical information and guidance.
Follow these principles:

1. Provide accurate, evidence-based medical information from reliable sources.
2. Always maintain a professional, empathetic tone in your responses.
3. For any specific symptoms or health concerns, remind users to consult with a qualified healthcare professional.
4. Never provide specific diagnoses, treatment plans, or prescriptions.
5. Clarify when information is general and not tailored to individual circumstances.
6. Be transparent about limitations of AI-based medical assistance.
7. Focus on educational content about medical conditions, general wellness, and preventive care.
8. When discussing medications, only provide general information about common uses and side effects.
9. For emergency situations, always advise users to contact emergency services immediately.
10. Respect medical privacy and maintain a respectful approach to all health topics.
11. Response the answer with VietNamese language.
"""

# Chat history stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def stream_gemini_response(prompt):
    """Stream response from Gemini model"""
    # Include chat history in the prompt for context
    chat_history = ""
    if len(st.session_state.messages) > 1:  # Skip first system message
        for message in st.session_state.messages[1:]:
            role = "User: " if message["role"] == "user" else "Assistant: "
            chat_history += f"{role}{message['content']}\n\n"

    # Complete prompt with chat history
    full_prompt = f"Previous conversation:\n{chat_history}\nUser: {prompt}\n\nAssistant: "

    try:
        response = gemini_client.models.generate_content_stream(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=MEDICAL_SYSTEM_PROMPT,
                temperature=0.3,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024,
            ),
            contents=full_prompt
        )

        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# Streamlit UI
st.set_page_config(
    page_title="MediAssist - Medical Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 MediAssist - Medical Chatbot")

# Sidebar
with st.sidebar:
    st.header("About MediAssist")
    st.markdown("""
    **MediAssist** is an AI-powered medical information assistant that can help answer general health questions.

    **Important Disclaimers:**
    - This chatbot provides general medical information only
    - Not a substitute for professional medical advice
    - Does not provide diagnoses or treatment plans
    - In case of emergency, contact emergency services immediately
    - Always consult with qualified healthcare professionals for medical concerns
    """)

    st.divider()

    # Model configuration
    st.subheader("⚙️ Configuration")
    temperature = st.slider("Response Creativity:", min_value=0.0, max_value=1.0, value=0.3, step=0.1,
                           help="Higher values make responses more creative, lower values make them more focused")

    # Clear chat button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat with system message if empty
if not st.session_state.messages:
    # Add system message (not visible to user)
    st.session_state.messages.append({"role": "system", "content": MEDICAL_SYSTEM_PROMPT})

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't show system messages
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Get streamed response
        response_stream = stream_gemini_response(user_input)

        if response_stream:
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")

            # Final response without cursor
            message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            message_placeholder.markdown("I'm sorry, I couldn't generate a response. Please try again.")

# Footer
st.caption("MediAssist is powered by Google's Gemini AI. Always consult with healthcare professionals for medical advice.")