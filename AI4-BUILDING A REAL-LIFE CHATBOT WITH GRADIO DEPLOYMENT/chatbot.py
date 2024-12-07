import speech_recognition as sr
import pyttsx3
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
# import gradio as gr

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set speech rate (speed of speech)
engine.setProperty('rate', 150)

# Function for text-to-speech output
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to capture speech and convert to text using Speech Recognition
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return ""
        except sr.RequestError:
            print("Sorry, the service is down.")
            return ""

# Define the LLM chain
llm = Ollama(model="qwen2.5-coder:0.5b")
output_parser = StrOutputParser()

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question:{question}")
    ]
)

# Function to handle the complete chatbot process
def chatbot():
    while True:
        # Step 1: Get speech input from the user
        input_text = speech_to_text()

        # If no valid input, continue listening
        if not input_text:
            continue

        # Step 2: Generate response from the LLM
        chain = prompt | llm | output_parser
        answer = chain.invoke({"question": input_text})
        print(f"Answer: {answer}")

        # Step 3: Convert the answer to speech and speak it
        speak(answer)

# Run the chatbot
if __name__ == "__main__":
    chatbot()