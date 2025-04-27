import streamlit as st
import os
import logging
import time
import json
from dotenv import load_dotenv
import openai
import numpy as np
from typing import Dict, List, Any, Optional

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
    openai.api_key = api_key

# ================ SIMPLIFIED AI AGENT COMPONENTS ================

# NLU Module
class NLUModule:
    def __init__(self):
        """Initialize the NLU module."""
        self.logger = logging.getLogger('nlu_module')
    
    def process_input(self, text):
        """
        Process user input to extract intent, entities, and sentiment.
        
        Args:
            text (str): The user input text
            
        Returns:
            dict: A dictionary containing processed information
        """
        # Basic preprocessing
        processed_text = text.strip()
        
        # Extract intent (simplified version)
        intent_result = self.classify_intent(processed_text)
        
        # Extract entities (simplified)
        entities = self.extract_entities(processed_text)
        
        # Analyze sentiment (simplified)
        sentiment = self.analyze_sentiment(processed_text)
        
        return {
            "processed_text": processed_text,
            "intent": intent_result,
            "entities": entities,
            "sentiment": sentiment
        }
    
    def classify_intent(self, text):
        """
        Classify the intent of the user input.
        
        Args:
            text (str): The user input text
            
        Returns:
            str: The classified intent
        """
        # Simplified rule-based approach
        text_lower = text.lower()
        
        if any(q in text_lower for q in ["what", "who", "when", "where", "why", "how"]):
            return "question"
        elif any(cmd in text_lower for cmd in ["find", "search", "look up", "get"]):
            return "search"
        elif any(cmd in text_lower for cmd in ["create", "make", "build", "generate"]):
            return "create"
        elif any(cmd in text_lower for cmd in ["help", "assist", "support"]):
            return "help"
        else:
            return "statement"
    
    def extract_entities(self, text):
        """
        Extract named entities from the user input (simplified).
        
        Args:
            text (str): The user input text
            
        Returns:
            dict: A dictionary of extracted entities
        """
        # Simplified entity extraction
        entities = {}
        
        # Look for dates
        date_indicators = ["today", "tomorrow", "yesterday", "next week", "last week"]
        for indicator in date_indicators:
            if indicator in text.lower():
                if "DATE" not in entities:
                    entities["DATE"] = []
                entities["DATE"].append({"word": indicator, "score": 0.9})
        
        # Look for locations
        location_indicators = ["in", "at", "near", "from"]
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in location_indicators and i < len(words) - 1:
                if "LOCATION" not in entities:
                    entities["LOCATION"] = []
                entities["LOCATION"].append({"word": words[i+1], "score": 0.8})
        
        return entities
    
    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the user input (simplified).
        
        Args:
            text (str): The user input text
            
        Returns:
            dict: The sentiment analysis result
        """
        # Simplified sentiment analysis
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy", "like", "love"]
        negative_words = ["bad", "terrible", "awful", "horrible", "sad", "hate", "dislike"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {"label": "POSITIVE", "score": 0.8}
        elif negative_count > positive_count:
            return {"label": "NEGATIVE", "score": 0.8}
        else:
            return {"label": "NEUTRAL", "score": 0.9}

# Conversation Manager
class ConversationManager:
    def __init__(self, max_history_length=20):
        """
        Initialize the conversation manager.
        
        Args:
            max_history_length (int): Maximum number of turns to keep in history
        """
        self.conversation_history = []
        self.current_context = {}
        self.max_history_length = max_history_length
        self.session_start_time = time.time()
        
    def add_user_message(self, message: str, metadata: Dict[str, Any] = None):
        """
        Add a user message to the conversation history.
        
        Args:
            message (str): The user message
            metadata (dict, optional): Additional metadata
        """
        message_entry = {
            "role": "user",
            "content": message,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(message_entry)
        self._trim_history()
        
    def add_assistant_message(self, message: str, metadata: Dict[str, Any] = None):
        """
        Add an assistant message to the conversation history.
        
        Args:
            message (str): The assistant message
            metadata (dict, optional): Additional metadata
        """
        message_entry = {
            "role": "assistant",
            "content": message,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(message_entry)
        self._trim_history()
        
    def _trim_history(self):
        """
        Trim the conversation history to the maximum length.
        """
        if len(self.conversation_history) > self.max_history_length:
            # Remove oldest messages but keep pairs of user/assistant messages
            # to maintain conversation coherence
            excess = len(self.conversation_history) - self.max_history_length
            self.conversation_history = self.conversation_history[excess:]
            
    def get_conversation_history(self, include_metadata=False):
        """
        Get the conversation history.
        
        Args:
            include_metadata (bool): Whether to include metadata
            
        Returns:
            list: The conversation history
        """
        if include_metadata:
            return self.conversation_history
        else:
            return [
                {
                    "role": entry["role"],
                    "content": entry["content"]
                }
                for entry in self.conversation_history
            ]
            
    def update_context(self, key: str, value: Any):
        """
        Update the current conversation context.
        
        Args:
            key (str): Context key
            value (Any): Context value
        """
        self.current_context[key] = value
        
    def get_context(self, key: str = None):
        """
        Get the current conversation context.
        
        Args:
            key (str, optional): Specific context key to retrieve
            
        Returns:
            Any: The context value or entire context dictionary
        """
        if key is not None:
            return self.current_context.get(key)
        else:
            return self.current_context
        
    def clear_context(self):
        """
        Clear the current conversation context.
        """
        self.current_context = {}

# Response Generator
class ResponseGenerator:
    def __init__(self, api_key=None):
        """
        Initialize the response generator.
        
        Args:
            api_key (str, optional): OpenAI API key
        """
        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            
        self.system_prompt = """
        You are a helpful AI assistant designed to provide informative, accurate, and friendly responses.
        You can answer questions, provide information, and assist with various tasks.
        Always be respectful, avoid harmful content, and admit when you don't know something.
        """
    
    def generate_response(self, 
                         conversation_history: List[Dict[str, str]], 
                         context: Dict[str, Any] = None):
        """
        Generate a response based on conversation history and context.
        
        Args:
            conversation_history (list): List of conversation messages
            context (dict, optional): Additional context information
            
        Returns:
            str: The generated response
        """
        try:
            # Prepare messages for the API
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history
            for message in conversation_history:
                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
            
            # Add context as a system message if provided
            if context:
                context_str = "Additional context:\n"
                for key, value in context.items():
                    context_str += f"- {key}: {value}\n"
                messages.append({"role": "system", "content": context_str})
            
            # Generate response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please check your OpenAI API key and try again."

# Simplified AI Agent
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
            conv_status = st.empty()
            resp_status = st.empty()
            
            nlu_status.info("Initializing NLU module...")
            self.nlu = NLUModule()
            nlu_status.success("NLU module initialized")
            
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
                "content": "I apologize, but I encountered an error while processing your message. Please check your OpenAI API key and try again.",
                "error": str(e)
            }

# ================ MAIN APPLICATION LOGIC ================

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
