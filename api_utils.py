'''
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from groq import Groq
import os

load_dotenv(find_dotenv())
api_key=os.environ['OPENAI_API_KEY']
client=OpenAI(api_key=api_key)
client_groq=Groq(api_key=os.environ["GROQ_API_KEY"])
'''

def openai_embed(client,text):
    response=client.embeddings.create(
        input=text,
        model="text-embedding-ada-002" #text-embedding-3-small
    )
    return response.data[0].embedding

def audio_to_text(audio_file_path, client):
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    text_content = transcript.text
    return text_content

def text_to_audio(text,client,output_path):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(output_path)
        return output_path
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

async def chat_completion(conversation_history, client_groq):
    response = await client_groq.chat.completions.create(
        messages=conversation_history,
        model="llama3-8b-8192",
        max_tokens=200
    )

    response_text = response.choices[0].message.content
    return response_text