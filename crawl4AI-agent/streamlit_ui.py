from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import uuid


def load_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from pydantic_ai_expert import get_agent, PydanticAIDeps, SYSTEM_PROMPTS

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

        # async def run_agent_with_streaming(user_input: str):


#     """
#     Run the agent with streaming text for the user_input prompt,
#     while maintaining the entire conversation in `st.session_state.messages`.
#     """
#     # Prepare dependencies
#     deps = PydanticAIDeps(
#         supabase=supabase,
#         openai_client=openai_client
#     )
# 
#     # Run the agent in a stream
#     async with pydantic_ai_expert.run_stream(
#         user_input,
#         deps=deps,
#         message_history= st.session_state.messages[:-1],  # pass entire conversation so far
#     ) as result:
#         # We'll gather partial text to show incrementally
#         partial_text = ""
#         message_placeholder = st.empty()
# 
#         # Render partial text as it arrives
#         async for chunk in result.stream_text(delta=True):
#             partial_text += chunk
#             message_placeholder.markdown(partial_text)
# 
#         # Now that the stream is finished, we have a final result.
#         # Add new messages from this run, excluding user-prompt messages
#         filtered_messages = [msg for msg in result.new_messages() 
#                             if not (hasattr(msg, 'parts') and 
#                                     any(part.part_kind == 'user-prompt' for part in msg.parts))]
#         st.session_state.messages.extend(filtered_messages)
# 
#         # Add the final response to the messages
#         st.session_state.messages.append(
#             ModelResponse(parts=[TextPart(content=partial_text)])
#         )
# async def run_agent_with_streaming(user_input: str, system_prompt_key: str):
#     """
#     Run the agent with streaming text for the user_input prompt,
#     while maintaining the entire conversation in `st.session_state.messages`.
#     """
#     # Get the selected system prompt
#     selected_system_prompt = SYSTEM_PROMPTS[system_prompt_key]
# 
#     # Prepare dependencies
#     deps = PydanticAIDeps(
#         supabase=supabase,
#         openai_client=openai_client
#     )
# 
#     # Create a new instance of the agent with the selected system prompt
#     agent = pydantic_ai_expert.with_system_prompt(selected_system_prompt)
# 
#     # Run the agent in a stream
#     async with agent.run_stream(
#             user_input,
#             deps=deps,
#             message_history=st.session_state.messages[:-1],  # pass entire conversation so far
#     ) as result:
#         # We'll gather partial text to show incrementally
#         partial_text = ""
#         message_placeholder = st.empty()
# 
#         # Render partial text as it arrives
#         async for chunk in result.stream_text(delta=True):
#             partial_text += chunk
#             message_placeholder.markdown(partial_text)
# 
#         # Now that the stream is finished, we have a final result.
#         # Add new messages from this run, excluding user-prompt messages
#         filtered_messages = [msg for msg in result.new_messages()
#                              if not (hasattr(msg, 'parts') and
#                                      any(part.part_kind == 'user-prompt' for part in msg.parts))]
#         st.session_state.messages.extend(filtered_messages)
# 
#         # Add the final response to the messages
#         st.session_state.messages.append(
#             ModelResponse(parts=[TextPart(content=partial_text)])
#         )

# In streamlit_ui.py

# Update the import statement
from pydantic_ai_expert import get_agent, PydanticAIDeps, SYSTEM_PROMPTS


# Then modify the run_agent_with_streaming function
async def run_agent_with_streaming(user_input: str, system_prompt_key: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Get the agent with the selected system prompt
    agent = get_agent(system_prompt_key)

    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    # Run the agent in a stream
    async with agent.run_stream(
            user_input,
            deps=deps,
            message_history=st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages()
                             if not (hasattr(msg, 'parts') and
                                     any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

import json
import uuid
from fastapi.encoders import jsonable_encoder


def save_conversation():
    # Ensure conversation_id is initialized
    if "conversation_id" not in st.session_state or not st.session_state.conversation_id:
        st.session_state.conversation_id = str(uuid.uuid4())  # Ensure we always have an ID

    if not st.session_state.messages:
        return

    # Extract title from the first message
    first_message = st.session_state.messages[0]
    title = (
        first_message.parts[0].content[:50]
        if first_message.parts and hasattr(first_message.parts[0], "content")
        else "New Conversation"
    )

    # Serialize messages to JSON
    serialized_messages = jsonable_encoder(st.session_state.messages)

    # Filter the serialized messages to include 'text' AND 'user_prompt' parts
    filtered_messages = []
    for msg in serialized_messages:
        # Filter out parts where part_kind is not 'text' or 'user_prompt'
        filtered_parts = [part for part in msg.get('parts', [])
                          if part.get('part_kind') in ['text', 'user-prompt']]

        # Only include messages that have parts after filtering
        if filtered_parts:
            msg['parts'] = filtered_parts
            filtered_messages.append(msg)

    # Save conversation in Supabase
    supabase.table("conversations").upsert({
        "id": st.session_state.conversation_id,  # Ensure UUID is set
        "title": title,
        "messages": json.dumps(filtered_messages)
    }).execute()


def load_conversation(conversation_id):
    response = supabase.table("conversations").select("messages").eq("id", conversation_id).execute()
    if response.data:
        raw_messages = json.loads(response.data[0]["messages"])

        # Reset messages
        st.session_state.messages = []

        # Reconstruct message objects from JSON data
        for msg in raw_messages:
            kind = msg.get("kind")
            parts_data = msg.get("parts", [])

            # Reconstruct parts objects
            parts = []
            for part_data in parts_data:
                part_kind = part_data.get("part_kind")
                content = part_data.get("content", "")

                if part_kind == "user-prompt":
                    parts.append(UserPromptPart(content=content))
                elif part_kind == "text":
                    parts.append(TextPart(content=content))
                elif part_kind == "retry-prompt":
                    parts.append(RetryPromptPart(content=content))

            # Create appropriate message object
            if kind == "request":
                st.session_state.messages.append(ModelRequest(parts=parts))
            elif kind == "response":
                st.session_state.messages.append(ModelResponse(parts=parts))
            # Add other message types if needed

        # Set conversation ID
        st.session_state.conversation_id = conversation_id
        st.rerun()

def list_conversations():
    response = supabase.table("conversations").select("id", "title", "created_at").order("created_at",
                                                                                         desc=True).execute()
    return response.data if response.data else []


# async def main():
#     st.title("Velkommen til Lægehåndbogens AI-assistent")
#     st.write("Du kan stille mig spørgsmål om sygdomme og behandlinger, og jeg vil forsøge at hjælpe dig, og give referencer til supplerende viden.")
# 
#     # Load custom CSS
#     load_css("styles.css")
# 
#     # Initialize chat history in session state if not present
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
# 
#     # Display all messages from the conversation so far
#     # Each message is either a ModelRequest or ModelResponse.
#     # We iterate over their parts to decide how to display them.
#     for msg in st.session_state.messages:
#         if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
#             for part in msg.parts:
#                 display_message_part(part)
# 
#     # Chat input for the user
#     user_input = st.chat_input("Hvad kunne du godt tænke dig at vide?")
# 
#     if user_input:
#         # We append a new request to the conversation explicitly
#         st.session_state.messages.append(
#             ModelRequest(parts=[UserPromptPart(content=user_input)])
#         )
#         
#         # Display user prompt in the UI
#         with st.chat_message("user"):
#             st.markdown(user_input)
# 
#         # Display the assistant's partial response while streaming
#         with st.chat_message("assistant"):
#             # Actually run the agent now, streaming the text
#             await run_agent_with_streaming(user_input)
# async def main():
#     st.title("Velkommen til Lægehåndbogens AI-assistent")
# 
#     # Add dropdown menu for selecting system prompts
#     system_prompt_key = st.sidebar.selectbox(
#         "Vælg assistentens rolle:",
#         options=list(SYSTEM_PROMPTS.keys()),
#         index=0  # Default to the first prompt
#     )
# 
#     # Display a description of the selected role
#     st.sidebar.markdown(f"**Valgt rolle:** {system_prompt_key}")
# 
#     st.write("Du kan stille mig spørgsmål om sygdomme og behandlinger, og jeg vil forsøge at hjælpe dig, og give referencer til supplerende viden.")
# 
#     # Load custom CSS
#     load_css("styles.css")
# 
#     # Initialize chat history in session state if not present
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
# 
#     # Reset chat history if the system prompt changes
#     if "current_system_prompt" not in st.session_state:
#         st.session_state.current_system_prompt = system_prompt_key
# 
#     if st.session_state.current_system_prompt != system_prompt_key:
#         st.session_state.messages = []
#         st.session_state.current_system_prompt = system_prompt_key
#         st.rerun()
# 
#     # Display all messages from the conversation so far
#     for msg in st.session_state.messages:
#         if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
#             for part in msg.parts:
#                 display_message_part(part)
# 
#     # Chat input for the user
#     user_input = st.chat_input("Hvad kunne du godt tænke dig at vide?")
# 
#     if user_input:
#         # We append a new request to the conversation explicitly
#         st.session_state.messages.append(
#             ModelRequest(parts=[UserPromptPart(content=user_input)])
#         )
# 
#         # Display user prompt in the UI
#         with st.chat_message("user"):
#             st.markdown(user_input)
# 
#         # Display the assistant's partial response while streaming
#         with st.chat_message("assistant"):
#             # Actually run the agent now, streaming the text
#             await run_agent_with_streaming(user_input, system_prompt_key)

async def main():
    st.title("Velkommen til Lægehåndbogens AI-assistent")

    # Add dropdown menu for selecting system prompts
    system_prompt_key = st.sidebar.selectbox(
        "Vælg assistentens rolle:",
        options=list(SYSTEM_PROMPTS.keys()),
        index=0  # Default to the first prompt
    )

    # Display a description of the selected role
    st.sidebar.markdown(f"**Valgt rolle:** {system_prompt_key}")

    st.write(
        "Du kan stille mig spørgsmål om sygdomme og behandlinger, og jeg vil forsøge at hjælpe dig, og give referencer til supplerende viden.")

    # Load custom CSS
    load_css("styles.css")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Reset chat history if the system prompt changes
    if "current_system_prompt" not in st.session_state:
        st.session_state.current_system_prompt = system_prompt_key

    if st.session_state.current_system_prompt != system_prompt_key:
        st.session_state.messages = []
        st.session_state.current_system_prompt = system_prompt_key
        st.rerun()

    # Display all messages from the conversation so far
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("Hvad kunne du godt tænke dig at vide?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input, system_prompt_key)
            save_conversation()

    # Sidebar UI
    st.sidebar.header("Tidligere samtaler")
    conversations = list_conversations()

    for convo in conversations:
        if st.sidebar.button(convo["title"], key=convo["id"]):
            load_conversation(convo["id"])

    # Button to start a new chat
    if st.sidebar.button("Start ny samtale"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()


if __name__ == "__main__":
    asyncio.run(main())
