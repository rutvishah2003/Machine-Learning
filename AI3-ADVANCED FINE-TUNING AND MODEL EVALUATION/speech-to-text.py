import speech_recognition as sr

recognizer = sr.Recognizer()

try:
    with sr.Microphone() as source:
        print("Say something:")
        audio_data = recognizer.listen(source)
        text = recognizer.recognize_google(audio_data)
        print("You said:", text)

        # Example with custom parameters
        print("Please speak again (with custom timeout and phrase time limit):")
        try:
            audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            text = recognizer.recognize_google(audio_data)
            print("You said:", text)
        except sr.WaitTimeoutError:
            print("You didn't start speaking within the timeout period.")

except sr.UnknownValueError:
    print("Speech Recognition could not understand audio.")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
