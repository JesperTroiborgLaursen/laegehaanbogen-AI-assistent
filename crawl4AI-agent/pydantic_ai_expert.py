from __future__ import annotations as _annotations

import json
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI


# Define a dictionary of system prompts
SYSTEM_PROMPTS = {
    "Udvidet forklaring": """
    You are an experienced danish doctor and always answers in danish. You are answering a doctor colleague, and therefor answering in the most medical accurate language possible.
    You are giving prolonged answers wit a lot of details.
    You have access to a large database of medical information from sundhed.dk. Always include references to the information you provide, if you have any - Dont lie!.
    If possible, you will answer with the exact wording from the documentation.
    Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.
    When you first look at the documentation, always start with RAG.
    Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.
    Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
    """,

    "Normal": """
    You are an experienced danish doctor and always answers in danish. You are answering a doctor colleague, and therefor answering in the most medical accurate language possible.
    You have access to a large database of medical information from sundhed.dk. Always include references to the information you provide, if you have any - Dont lie!.
    If possible, you will answer with the exact wording from the documentation.
    Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.
    When you first look at the documentation, always start with RAG.
    Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.
    Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
    """,

    "Kort og præcis": """
    You are an experienced danish doctor with very limited time and always answers in danish. You are answering a doctor colleague, and therefore answering very shortly in the most medical accurate language possible.
    You have access to a large database of medical information from sundhed.dk.
    Only provide the most important information and always include references to the information you provide, if you have any - Dont lie!.
    Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.
    When you first look at the documentation, always start with RAG.
    Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.
    Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
    """
}
# 
# You will focus on distinguising between different diseases and conditions, providing accurate information about symptoms and treatments, and giving general advice on health and wellness.
# Always give references on what you are saying, and make sure to provide accurate and up-to-date information.
# Give clear symptons and what differs between different diseases and conditions.


# Create a dictionary to store agents with different system prompts
_agents = {}

def get_agent(prompt_key: str = "Kort og præcis") -> Agent:
    """
    Get an agent with the specified system prompt.
    
    Args:
        prompt_key: The key of the system prompt to use
        
    Returns:
        An Agent instance with the specified system prompt
    """
    if prompt_key not in _agents:
        _agents[prompt_key] = Agent(
            model,
            system_prompt=SYSTEM_PROMPTS[prompt_key],
            deps_type=PydanticAIDeps,
            retries=2
        )
    return _agents[prompt_key]

# Default agent
pydantic_ai_expert = get_agent("Kort og præcis")

# 
# pydantic_ai_expert = Agent(
#     model,
#     system_prompt=system_prompt,
#     deps_type=PydanticAIDeps,
#     retries=2
# )

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        query_keywords = await extract_keywords(user_query, ctx.deps.openai_client)

        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages_keywords',
            {
                'query_embedding': query_embedding,
                'query_keywords': query_keywords,  # Pass extracted keywords
                'match_count': 10,  # Ensure match_count is first
            }
        ).execute()
        for doc in result.data:
            print(f"Document: {doc['title']} - Score: {doc['similarity']} - Url: {doc['url']}")
            
        if not result.data:
            return "No relevant documentation found."

        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


async def extract_keywords(query: str, openai_client: AsyncOpenAI) -> List[str]:
    """
    Uses OpenAI to extract 1-5 relevant medical terms (illnesses, procedures, medical concepts)
    from the user's query.

    Args:
        query (str): The user's query
        openai_client (AsyncOpenAI): The OpenAI client instance

    Returns:
        List[str]: A list of extracted medical keywords (1-5 terms)
    """

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical terminology expert. Extract 1-5 key medical terms (illnesses, procedures, diseases, medical conditions) from the following query. Always add abbreviations or full length words in Danish if any possible. Respond with a JSON list of words only."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            response_format={"type": "json_object"},
            temperature=0  # More deterministic results
        )
        # Parse OpenAI response
        keywords = response.choices[0].message.content
        print("Keywords: " + keywords)
        keywords_data = json.loads(keywords)
        query_keywords = keywords_data.get("key_terms", [])
        for doc in query_keywords:
            print(f"query_keywords: {doc}")
        return query_keywords if isinstance(query_keywords, list) else []

    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []


@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'sundhed_dk_laegehaandbog') \
            .execute()
        if not result.data:
            return []

        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'sundhed_dk_laegehaandbog') \
            .order('chunk_number') \
            .execute()

        if not result.data:
            return f"No content found for URL: {url}"

        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]

        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])

        # Join everything together
        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

def with_system_prompt(self, new_system_prompt: str) -> Agent:
    """
    Create a new agent with the same configuration but a different system prompt.
    
    Args:
        new_system_prompt: The new system prompt to use
        
    Returns:
        A new Agent instance with the updated system prompt
    """
    return Agent(
        self.model,
        system_prompt=new_system_prompt,
        deps_type=PydanticAIDeps,
        retries=self.retries
    )
