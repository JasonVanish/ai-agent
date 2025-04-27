import streamlit as st
import os
import sys
import logging
from dotenv import load_dotenv

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the AI agent components
from src.nlu import NLUModule
from src.knowledge_base import KnowledgeBase
from src.task_execution import TaskExecutionEngine
from src.conversation import ConversationManager
from src.response_generator import ResponseGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ai_agent_streamlit')

# Load environment variables from .env file if present
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("AI Agent")
st.markdown("""
This is an AI agent with conversational capabilities similar to advanced AI assistants.
It can understand natural language, process information, and maintain context throughout conversations.
""")

# Sidebar with API key input
st.sidebar.title("Configuration")
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:",
    value=st.session_state.openai_api_key,
    type="password",
    help="You can get an API key from https://platform.openai.com/api-keys"
)

if api_key:
    st.session_state.openai_api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key

# Simplified AIAgent class for Streamlit
class StreamlitAIAgent:
    def __init__(self, api_key=None):
        """
        Initialize the AI agent with all its components.
        
        Args:
            api_key (str, optional): OpenAI API key
        """
        # Initialize components with progress indicators
        with st.sidebar.expander("Agent Initialization Status", expanded=True):
            nlu_status = st.empty()
            kb_status = st.empty()
            task_status = st.empty()
            conv_status = st.empty()
            resp_status = st.empty()
            
            nlu_status.info("Initializing NLU module...")
            self.nlu = NLUModule()
            nlu_status.success("NLU module initialized")
            
            kb_status.info("Initializing knowledge base...")
            self.knowledge_base = KnowledgeBase(api_key=api_key)
            kb_status.success("Knowledge base initialized")
            
            task_status.info("Initializing task execution engine...")
            self.task_engine = TaskExecutionEngine()
            task_status.success("Task execution engine initialized")
            
            conv_status.info("Initializing conversation manager...")
            self.conversation = ConversationManager()
            conv_status.success("Conversation manager initialized")
            
            resp_status.info("Initializing response generator...")
            self.response_generator = ResponseGenerator(api_key=api_key)
            resp_status.success("Response generator initialized")
        
        st.sidebar.success("AI agent initialization complete")
    
    def process_message(self, message, metadata=None):
        """
        Process a user message and generate a response.
        
        Args:
            message (str): The user message
            metadata (dict, optional): Additional metadata
            
        Returns:
            dict: The agent's response
        """
        try:
            logger.info(f"Processing message: {message[:50]}...")
            
            # Add message to conversation history
            self.conversation.add_user_message(message, metadata)
            
            # Process with NLU
            nlu_result = self.nlu.process_input(message)
            logger.info(f"NLU result: Intent={nlu_result['intent']}")
            
            # Update conversation context with NLU results
            self.conversation.update_context("intent", nlu_result["intent"])
            self.conversation.update_context("entities", nlu_result["entities"])
            
            # Handle based on intent
            if nlu_result["intent"] == "question":
                # Search knowledge base
                kb_results = self.knowledge_base.search(message)
                self.conversation.update_context("knowledge_results", kb_results)
                
            elif nlu_result["intent"] == "search":
                # Execute search task
                logger.info("Executing search task...")
                # This would call a specific search function in the task engine
                
            elif nlu_result["intent"] == "create":
                # Execute creation task
                logger.info("Executing creation task...")
                # This would call a specific creation function in the task engine
            
            # Get conversation history for response generation
            history = self.conversation.get_conversation_history()
            context = self.conversation.get_context()
            
            # Generate response
            response_text = self.response_generator.generate_response(history, context)
            
            # Add response to conversation history
            self.conversation.add_assistant_message(response_text)
            
            return {
                "content": response_text,
                "metadata": {
                    "intent": nlu_result["intent"],
                    "sentiment": nlu_result["sentiment"]["label"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "content": "I apologize, but I encountered an error while processing your message. Please try again.",
                "error": str(e)
            }

# Main application logic
if st.session_state.openai_api_key:
    # Initialize the agent if not already done
    if "agent" not in st.session_state:
        with st.spinner("Initializing AI agent..."):
            st.session_state.agent = StreamlitAIAgent(api_key=st.session_state.openai_api_key)
            st.session_state.messages = []
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.process_message(prompt)
                st.markdown(response["content"])
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["content"]})
    
    # Sidebar with conversation information
    with st.sidebar.expander("Conversation Information", expanded=False):
        if len(st.session_state.messages) > 0:
            if hasattr(st.session_state.agent, 'conversation') and hasattr(st.session_state.agent.conversation, 'get_context'):
                context = st.session_state.agent.conversation.get_context()
                st.json(context)
        else:
            st.info("Start a conversation to see information here.")
    
    # Add a button to clear the conversation
    if st.sidebar.button("Clear Conversation"):
        st.session_state.messages = []
        if hasattr(st.session_state.agent, 'conversation') and hasattr(st.session_state.agent.conversation, 'clear_context'):
            st.session_state.agent.conversation.clear_context()
        st.experimental_rerun()
        
else:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    st.info("You can get an API key from https://platform.openai.com/api-keys")
    
    # Example conversation
    st.subheader("Example Conversation")
    with st.chat_message("user"):
        st.markdown("What can you help me with?")
    
    with st.chat_message("assistant"):
        st.markdown("""
        I can help you with a variety of tasks, including:
        
        1. Answering questions and providing information
        2. Helping you understand complex topics
        3. Assisting with problem-solving
        4. Maintaining context throughout our conversation
        
        Feel free to ask me anything, and I'll do my best to assist you!
        """)
