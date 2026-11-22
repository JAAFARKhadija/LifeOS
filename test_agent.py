
from modules.ai_agent import chat_with_lifeos

print("Testing LifeOS AI Agent...")

response = chat_with_lifeos(
    "Give me an overview of my pending tasks and their deadlines."
)

print("\nLIFEOS AI RESPONSE:")
print(response)
