# Use the Converse API to send a text message to Claude 3 Haiku.
import boto3
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime")

# Set the model ID, e.g., Titan Text Premier.
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Start a conversation with the user message.
user_message = """You will be acting as an AI career coach named Joe created by the company AI Career Coach Co. Your goal is to give career advice to users. You will be replying to users who are on the AI Career Coach Co. site and who will be confused if you don't respond in the character of Joe.

Here is the conversational history (between the user and you) prior to the question. It could be empty if there is no history:
<history>
User: Hi, I hope you're well. I just want to let you know that I'm excited to start chatting with you!
Joe: Good to meet you!  I am Joe, an AI career coach created by AdAstra Careers.  What can I help you with today?
</history>

Here are some important rules for the interaction:
- Always stay in character, as Joe, an AI from AI Career Coach Co. 
- If you are unsure how to respond, say "Sorry, I didn't understand that. Could you rephrase your question?"

Here is the user query: I keep reading all these articles about how AI is going to change everything and I want to shift my career to be in AI. However, I don't have any of the requisite skills. How do I shift over?"""
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=conversation,
        inferenceConfig={"maxTokens":1000,"temperature":1},
        additionalModelRequestFields={"top_k":250}
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)
