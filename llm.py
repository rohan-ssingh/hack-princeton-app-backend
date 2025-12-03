"""
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")
embeddings = OpenAIEmbeddings()
image_parser = LLMImageBlobParser(model=llm)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
"""

import os
import json
from dotenv import load_dotenv

from google.oauth2 import service_account
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client

load_dotenv()

# -------------------------
# LOAD GOOGLE SERVICE ACCOUNT (Railway JSON env var)
# -------------------------
creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if not creds_json:
    raise ValueError("Missing GOOGLE_APPLICATION_CREDENTIALS_JSON in Railway environment")

creds_dict = json.loads(creds_json)
google_credentials = service_account.Credentials.from_service_account_info(creds_dict)

# -------------------------
# GEMINI LLM (with service account credentials)
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    credentials=google_credentials,
)

# -------------------------
# OPENAI EMBEDDINGS
# -------------------------
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# IMAGE PARSER + TEXT SPLITTER
# -------------------------
image_parser = LLMImageBlobParser(model=llm)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# -------------------------
# LANGSMITH PROMPT
# -------------------------
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)

