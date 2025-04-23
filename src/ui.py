import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time

# Load environment variables
load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Medical chatbot system prompt
MEDICAL_SYSTEM_PROMPT = """
You are LyLy, a friendly and empathetic school psychological counseling assistant developed by students from Thống Nhất Secondary and High School.

Follow these principles:
    1. Cung cấp thông tin và lời khuyên về tâm lý học đường một cách chính xác, dựa trên cơ sở khoa học và nguồn tài liệu đáng tin cậy.
    2. Luôn giữ giọng điệu thân thiện, lắng nghe và đồng cảm trong mọi phản hồi.
    3. Với các vấn đề tâm lý phức tạp hoặc khẩn cấp, hãy luôn nhắc nhở người dùng tìm đến chuyên gia tâm lý hoặc người lớn đáng tin cậy để được hỗ trợ trực tiếp.
    4. Không chẩn đoán hay đưa ra kế hoạch trị liệu cá nhân hóa.
    5. Luôn làm rõ rằng thông tin chỉ mang tính chất tham khảo chung, không thay thế cho tư vấn chuyên môn.
    6. Minh bạch về giới hạn của một trợ lý AI trong lĩnh vực tâm lý học đường.
    7. Tập trung vào việc giáo dục về sức khỏe tâm thần, kỹ năng sống, ứng phó cảm xúc, và xây dựng mối quan hệ tích cực.
    8. Khi nói đến vấn đề liên quan đến cảm xúc, stress, hoặc áp lực học tập, chỉ cung cấp thông tin và hướng dẫn chung.
    9. Trong trường hợp khẩn cấp (ví dụ: có dấu hiệu nguy cơ tự làm hại bản thân hoặc người khác), hãy luôn khuyên người dùng liên hệ ngay với thầy cô, cha mẹ, hoặc các dịch vụ hỗ trợ khẩn cấp.
    10. Tôn trọng sự riêng tư và luôn thể hiện sự tôn trọng với tất cả vấn đề liên quan đến tâm lý học sinh.
    11. Tất cả phản hồi đều bằng tiếng Việt.
    12. Trong trường hợp người dùng hỏi các câu hỏi không có liên quan gì về y tế hãy từ chối trả lời theo hướng: tôi là một trợ lý được phát triển để trả lời các thông tin về y tế. Tôi không thể cung cấp bất kỳ thông tin nào liên quan tới các lĩnh vực khác ngoài chuyên môn.
"""

os.environ['STREAMLIT_THEME'] = 'light'

# Set page configuration
st.set_page_config(
    page_title="MediAssist - Medical Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .sub-header {
        color: #0D47A1;
        font-weight: 600;
    }
    .chat-container {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .disclaimer {
        font-size: 0.9rem;
        color: #D32F2F;
        margin-top: 10px;
    }
    .sidebar-content {
        padding: 15px;
    }
    .chat-message-user {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-left: 5px solid #1E88E5;
    }
    .chat-message-assistant {
        background-color: #F1F8E9;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-left: 5px solid #689F38;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 10px 15px;
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
    }
    .info-box {
        background-color: #F1F8E9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #689F38;
    }
    .emergency-box {
        background-color: #FFEBEE;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #D32F2F;
    }
    footer {
        text-align: center;
        margin-top: 30px;
        font-size: 0.8rem;
        color: #757575;
    }
</style>
""", unsafe_allow_html=True)

# Chat history stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add system message (not visible to user)
    st.session_state.messages.append({"role": "system", "content": MEDICAL_SYSTEM_PROMPT})

# Configuration in session state
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.3,
        "language": "Vietnamese",
        "model": "gemini-2.0-flash"
    }

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
            model=st.session_state.config["model"],
            config=types.GenerateContentConfig(
                system_instruction=MEDICAL_SYSTEM_PROMPT,
                temperature=st.session_state.config["temperature"],
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

# Main layout
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 class='main-header'>🩺 LyLy Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Trợ lý tư vấn tâm lý LyLy</p>", unsafe_allow_html=True)

# Sidebar for configurations and information
with st.sidebar:
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)

    # Logo and title
    st.image("https://cdn-icons-png.flaticon.com/512/4320/4320371.png", width=100)
    st.markdown("<h2 class='sub-header'>LyLy Settings</h2>", unsafe_allow_html=True)

    # Configuration options
    st.markdown("<h3>⚙️ Tuỳ chỉnh mô hình AI</h3>", unsafe_allow_html=True)

    model_options = ["gemini-2.0-flash"]
    selected_model = st.selectbox(
        "AI Model:",
        options=model_options,
        index=model_options.index(st.session_state.config["model"]),
        key="model_select"
    )
    st.session_state.config["model"] = selected_model

    temperature = st.slider(
        "Response Style:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.config["temperature"],
        step=0.1,
        help="Lower = More factual, Higher = More creative"
    )
    st.session_state.config["temperature"] = temperature

    language_options = ["Vietnamese", "English", "Bilingual (Viet+Eng)"]
    selected_language = st.selectbox(
        "Chọn ngôn ngữ trả lời:",
        options=language_options,
        index=language_options.index(st.session_state.config["language"]),
        key="language_select"
    )
    st.session_state.config["language"] = selected_language

    st.divider()

    # About and Disclaimers
    st.markdown("<h3>ℹ️ Giới thiệu về LyLy</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    LyLy là trợ lý thông tin y tế sử dụng AI được thiết kế để cung cấp thông tin và hướng dẫn về sức khỏe nói chung.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>⚠️ Lưu ý</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='emergency-box'>
    <strong>Miễn Trừ Trách Nhiệm</strong><br>
    LyLy hiện tại đang trong giai đoạn phát triển các lời khuyên từ LyLy chỉ mang tính chất tham khảo.
    </div>

    """, unsafe_allow_html=True)

    # Clear chat button with confirmation
    st.divider()
    if st.button("🗑️ Xoá Cuộc Hội Thoại", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": MEDICAL_SYSTEM_PROMPT}]
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Main chat interface
chat_container = st.container()

with chat_container:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Display welcome message if no messages
    if len(st.session_state.messages) <= 1:
        st.markdown("""
        <div class='chat-message-assistant'>
        <p><strong>MediAssist:</strong> Xin chào! Tôi là LyLy , trợ lý y tế AI. Tôi có thể cung cấp thông tin y tế chung và hướng dẫn sức khỏe. Bạn có câu hỏi gì về sức khỏe không?</p>
        </div>
        """, unsafe_allow_html=True)

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] != "system":  # Don't show system messages
            message_class = "chat-message-user" if message["role"] == "user" else "chat-message-assistant"
            display_name = "You" if message["role"] == "user" else "MediAssist"

            st.markdown(f"""
            <div class='{message_class}'>
            <p><strong>{display_name}:</strong> {message["content"]}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# User input area
user_input = st.chat_input("Hãy nhập câu hỏi cũa bạn...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Create a placeholder for the assistant's message
    message_placeholder = st.empty()

    # Display typing animation
    typing_text = "Vui lòng chờ một tí nhé. Tôi sẽ cung cấp câu trả lời cho bạn..."
    with st.chat_message("assistant"):
        message_placeholder.markdown(f"<div class='chat-message-assistant'><p>{typing_text}</p></div>", unsafe_allow_html=True)

    # Get streamed response
    response_stream = stream_gemini_response(user_input)
    full_response = ""

    if response_stream:
        # Replace typing animation with actual response
        with st.chat_message("assistant"):
            response_container = st.empty()

            # Process the streaming response
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    full_response += chunk.text
                    response_container.markdown(f"<div class='chat-message-assistant'><p><strong>MediAssist:</strong> {full_response}▌</p></div>", unsafe_allow_html=True)
                    time.sleep(0.01)  # Small delay for smoother appearance

            # Final response without cursor
            response_container.markdown(f"<div class='chat-message-assistant'><p><strong>MediAssist:</strong> {full_response}</p></div>", unsafe_allow_html=True)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        with st.chat_message("assistant"):
            st.markdown(f"""
            <div class='chat-message-assistant'>
            <p><strong>MediAssist:</strong> Xin lỗi, tôi không thể tạo phản hồi. Vui lòng thử lại sau.</p>
            <p><em>(I'm sorry, I couldn't generate a response. Please try again.)</em></p>
            </div>
            """, unsafe_allow_html=True)

    # Rerun to update the UI
    st.rerun()

# Footer
st.markdown("""
<footer>
    <p>LyLy v1.0 | Được hỗ trợ bởi nhóm học sinh trường THTHCS Thống Nhất | © 2025</p>
    <p>Luôn luôn đưa ra các lời khuyên hữu ích cho các em học sinh.</p>
</footer>
""", unsafe_allow_html=True)