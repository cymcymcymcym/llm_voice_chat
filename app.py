import os
import asyncio
import pyaudio
import wave
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from groq import AsyncGroq
import webrtcvad
from pydub import AudioSegment
import simpleaudio as sa
import sys
from api_utils import *

load_dotenv(find_dotenv(),override=True)
api_key = os.environ['OPENAI_API_KEY']
groq_api_key = os.environ["GROQ_API_KEY"]

client = OpenAI(api_key=api_key)
client_groq = AsyncGroq(api_key=groq_api_key)

def record_wav(timeout=2, silence_threshold=1):
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Aggressive mode

    form_1 = pyaudio.paInt16
    chans = 1
    samp_rate = 16000
    frame_duration = 10  # Frame duration in ms (10 ms for finer control)
    frame_size = int(samp_rate * frame_duration / 1000)
    chunk = frame_size * chans * 2
    wav_output_filename = 'input.wav'

    audio = pyaudio.PyAudio()
    stream = audio.open(format=form_1, rate=samp_rate, channels=chans, input=True, frames_per_buffer=chunk)
    
    frames = []
    sys.stdout.write("Listening...")
    sys.stdout.flush()

    silence_duration = 0
    is_speaking = False
    while True:
        data = stream.read(chunk)
        if vad.is_speech(data[:frame_size * 2], samp_rate):
            frames.append(data)
            silence_duration = 0
            if not is_speaking:
                sys.stdout.write("\rRecording...   ")
                sys.stdout.flush()
                is_speaking = True
        else:
            if is_speaking:
                silence_duration += frame_duration / 1000
                if silence_duration > silence_threshold:
                    sys.stdout.write("\rFinished recording\n")
                    sys.stdout.flush()
                    break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    if len(frames) == 0:
        return None

    wavefile = wave.open(wav_output_filename, 'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

    return wav_output_filename

def play_audio(file_path):
    try:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        audio = AudioSegment.from_file(file_path)
        play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing audio file: {e}")

async def main():
    conversation_history = [
        {"role": "system", "content": "You are an audio assistant. Use colloquial language and be concise in your responses. If the user is making casual comments, keep the response under 10 words. If the user is asking for technical and academic and emotional support, you may respond in 60 words or so."}
    ]

    while True:
        audio_file_path = record_wav()

        if audio_file_path is None:
            print("No speech detected. Skipping processing.")
            await asyncio.sleep(1)
            continue

        question = audio_to_text(audio_file_path, client)
        print(f"\033[34mUser:\033[0m {question}")

        conversation_history.append({"role": "user", "content": question})

        response_text = await chat_completion(conversation_history, client_groq)
        print("\033[32mModel:\033[0m", response_text)
        conversation_history.append({"role": "assistant", "content": response_text})

        response_audio_path = text_to_audio(response_text, client, "response.mp3")
        if response_audio_path:
            play_audio(response_audio_path)

        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
