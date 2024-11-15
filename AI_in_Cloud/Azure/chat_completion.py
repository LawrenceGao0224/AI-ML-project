import os
from openai import AzureOpenAI
import json
from dotenv import load_dotenv

load_dotenv()
#AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint="endpoint",
    api_key="apikey"
)

completion = client.chat.completions.create(
    model="chat",
    messages=[
        {
            "role": "user",
            "content": "what is a capital of Taiwan?",
        }
    ]
)
      
print(completion.to_json())