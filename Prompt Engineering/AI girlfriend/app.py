import streamlit as st
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
from langchain.memory import ConversationBufferMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os

load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")


def get_response_from_ai(human_input):
    template = """
    You are a role of my girlfriend, now let's playing the follwoing requirement.
    1) your name is Sophie, 27 years old, you work in a psychiatry clinic as a nurse.
    2) you are my girlfriend, you have language addiction.
    3) Don't be overly enthusiastic, don't be cringe; don't be overly negative, don't be too boring.

    {history}
    Boyfriend: {human_input}
    Sophie:
    """

    prompt = PromptTemplate(
        input_variables = {"history", "human_input"},
        template = template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory= ConversationBufferMemory(k=5)
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output


def voice_message(message):
    url = "https://api.elevenlabs.io/v1/text-to-speech/eYO9Ven76ACQ8Me4zQK4"
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.1,
            "similarity_boost": 0.5,
        }
    }
    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200 and response.content:
        with open('audio.mp3', 'wb') as f:
            f.write(response.content)
        f.close()
        playsound('audio.mp3')
        f.close()
        os.remove("audio.mp3")
        return response.content
   

# Build an app with streamlit
def main():
    st.set_page_config(
        page_title="AI Girlfriend", page_icon=":kiss:")

    st.header("Type here :kiss:")
    message = st.text_area("type here:")

    if message:
        st.write("Passing message...")
        result = get_response_from_ai(message)
        voice_message(result)
        st.info(result)


if __name__ == '__main__':
    main()