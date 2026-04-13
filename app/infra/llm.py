#llm.py
from openai import AzureOpenAI
import os

from dotenv import load_dotenv
load_dotenv()


client=AzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')

)

CHAT_MODEL=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")