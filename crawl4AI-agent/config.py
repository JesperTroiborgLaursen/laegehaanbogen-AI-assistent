"""Configuration and environment variables for the application."""
from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import Client, create_client

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = "text-embedding-3-small"

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# App Configuration
APP_TITLE = "Velkommen til Lægehåndbogens AI-assistent"
APP_DESCRIPTION = "Du kan stille mig spørgsmål om sygdomme og behandlinger, og jeg vil forsøge at hjælpe dig, og give referencer til supplerende viden."

# Configure clients
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Logfire configuration
import logfire
logfire.configure(send_to_logfire='if-token-present')