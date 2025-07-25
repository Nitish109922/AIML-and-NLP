import os
import re
import speech_recognition as sr
import pyttsx3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import datetime
import webbrowser
import tkinter as tk
from tkinter import scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
import subprocess
import requests
import sounddevice as sd
import soundfile as sf
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import python_speech_features as mfcc
import pickle
from scipy.io.wavfile import read
import warnings
import vosk
import json
import wave
from serpapi import GoogleSearch



SERPAPI_API_KEY = "dcc0f83ad70bf2ca76f20a693856e88bc1fb19efdf6bd343aa90af19be8a39c5"

warnings.filterwarnings("ignore")

# Initialize the recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Ensure necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Global variable to control the listening thread
listening = False

# Path to the trained speaker model
model_path = "Speakers_models/"

def speak(text):
    try:
        engine.setProperty('rate', 150)  # Adjust the rate (words per minute)
        engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)
        
        # Split text into smaller chunks
        chunks = re.split(r'(?<=[.!?]) +', text)
        for chunk in chunks:
            engine.say(chunk)
            engine.runAndWait()
    except Exception as e:
        print(f"Error in speech synthesis: {e}")
import requests

def check_internet():
    url = "http://www.google.com"
    timeout = 5
    try:
        requests.get(url, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

# Path to your Vosk model
vosk_model_path = "path/to/vosk-model-small-en-us-0.15"

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        
        if check_internet():
            # Use Google API for online recognition
            try:
                query = recognizer.recognize_google(audio)
                print(f"User said (Online): {query}")
                audio_file_path = "temp_audio.wav"
                with open(audio_file_path, "wb") as f:
                    f.write(audio.get_wav_data())
                return query, audio_file_path
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
                return "", None
            except sr.RequestError:
                print("Sorry, my speech service is down.")
                return "", None
        else:
            # Use Vosk for offline recognition
            try:
                model = vosk.Model(vosk_model_path)
                recognizer_vosk = vosk.KaldiRecognizer(model, 16000)
                audio_data = audio.get_wav_data()
                
                if recognizer_vosk.AcceptWaveform(audio_data):
                    result = recognizer_vosk.Result()
                    result_dict = json.loads(result)
                    query = result_dict.get("text", "")
                    print(f"User said (Offline): {query}")
                    return query, None
                else:
                    print("Sorry, I did not understand that.")
                    return "", None
            except Exception as e:
                print(f"Error in offline recognition: {e}")
                return "", None


def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""
    rows, cols = array.shape
    deltas = np.zeros((rows, cols))
    N = 2
    for i in range(rows):
        index = []
        for j in range(1, N + 1):
            first = max(0, i - j)
            second = min(rows - 1, i + j)
            index.append((second, first))
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas

def extract_features(audio, rate):
    """Extract 20-dim MFCC features, perform CMS, and combine with delta to make it a 40-dim feature vector."""
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined

def identify_speaker(audio_file_path):
    sr, audio = read(audio_file_path)
    vector = extract_features(audio, sr)
    
    gmm_files = [os.path.join(model_path, fname) for fname in os.listdir(model_path) if fname.endswith('.gmm')]
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]
    
    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    
    speaker = np.argmax(log_likelihood)
    return speakers[speaker]

def process_query(query, audio_file_path=None):
    if audio_file_path:
        # Identify the speaker
        try:
            speaker_name = identify_speaker(audio_file_path)
            speak(f"Hello {speaker_name}!")
            conversation_text.insert(tk.END, f"Assistant: Hello, {speaker_name}!\n")
        except Exception as e:
            print(f"Error identifying speaker: {e}")
            conversation_text.insert(tk.END, f"Assistant: Error identifying speaker.\n")

    tokens = word_tokenize(query.lower())
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

    if "time" in filtered_tokens:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        speak(f"The current time is {now}")
        conversation_text.insert(tk.END, f"Assistant: The current time is {now}\n")
    elif "search" in filtered_tokens:
        for i, word in enumerate(filtered_tokens):
            if word == "search":
                search_query = ' '.join(filtered_tokens[i + 1:])
                webbrowser.open(f"https://www.google.com/search?q={search_query}")
                speak(f"Searching for {search_query}")
                conversation_text.insert(tk.END, f"Assistant: Searching for {search_query}\n")
                break
    elif "open" in filtered_tokens:
        open_keyword_handler(filtered_tokens)
    elif "youtube" in filtered_tokens or "song" in filtered_tokens or "video" in filtered_tokens or "play" in filtered_tokens:
        filtered_tokens1 = [token for token in filtered_tokens if token.lower() != "play"]
        search_query = ' '.join(filtered_tokens1)
        search_youtube(search_query)
    elif any(word in filtered_tokens for word in ["who", "what", "when", "where", "why", "how"]):
        answer_question(query)
    elif "who" in filtered_tokens:
        answer_question(query)
    else:
        speak("I can only tell the time, search the web, open applications, or search YouTube for now.")
        conversation_text.insert(tk.END, "Assistant: I can only tell the time, search the web, open applications, or search YouTube for now.\n")

def answer_question(query):
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        # Try getting a direct answer box or snippet
        answer = results.get("answer_box", {}).get("answer") \
              or results.get("answer_box", {}).get("snippet") \
              or results.get("organic_results", [{}])[0].get("snippet")

        if answer:
            speak(answer)
            conversation_text.insert(tk.END, f"Assistant: {answer}\n")
        else:
            speak("I couldn't find a direct answer, but let me show you the search results.")
            webbrowser.open(f"https://www.google.com/search?q={query}")
            conversation_text.insert(tk.END, f"Assistant: Showing web results for: {query}\n")
    except Exception as e:
        print(f"Error in web search: {e}")
        conversation_text.insert(tk.END, f"Assistant: Web search failed.\n")

def open_keyword_handler(filtered_tokens):
    app_names = ["notepad", "calculator","youtube", "camera", "whatsapp", "calendar", "file explorer", "settings", "word", "excel", "powerpoint", "photos", "spotify", "prime video"]
    search_query = ' '.join(filtered_tokens)
    for app_name in app_names:
        if app_name in filtered_tokens:
            open_application(app_name)
            return
    # If no app name matches, treat it as a Google search
    webbrowser.open(f"https://www.google.com/search?q={search_query}")
    speak(f"Searching for {search_query}")
    conversation_text.insert(tk.END, f"Assistant: Searching for {search_query}\n")

def open_application(app_name):
    try:
        if app_name == "notepad":
            subprocess.Popen(["notepad.exe"])
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "calculator":
            subprocess.Popen(["calc.exe"])
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "camera":
            subprocess.Popen(["start", "microsoft.windows.camera:", "/C"], shell=True)
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "whatsapp":
            subprocess.Popen(["start", "whatsapp:", "/C"], shell=True)
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "calendar":
            subprocess.Popen(["start", "outlookcal:", "/C"], shell=True)
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "youtube":
            webbrowser.open("https://www.youtube.com")
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "file explorer":
            subprocess.Popen(["explorer.exe"])
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "settings":
            subprocess.Popen(["start", "ms-settings:", "/C"], shell=True)
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "word":
            subprocess.Popen([r'C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Word.lnk'], shell=True)
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "excel":
            subprocess.Popen([r'C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Excel.lnk'], shell=True)
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "powerpoint":
            subprocess.Popen([r'C:\ProgramData\Microsoft\Windows\Start Menu\Programs\PowerPoint.lnk'], shell=True)
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "photos":
            subprocess.Popen(["start", "microsoft.windows.photos:", "/C"], shell=True)
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "spotify":
            subprocess.Popen(["start", "spotify:", "/C"], shell=True)
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        elif app_name == "prime video":
            subprocess.Popen(["start", "primevideo:", "/C"], shell=True)
            speak(f"Opening {app_name}")
            conversation_text.insert(tk.END, f"Assistant: Opening {app_name}\n")
        else:
            speak(f"Sorry, I can't open {app_name} right now.")
            conversation_text.insert(tk.END, f"Assistant: Sorry, I can't open {app_name} right now.\n")
    except Exception as e:
        print(f"Error opening {app_name}: {e}")
        conversation_text.insert(tk.END, f"Assistant: Error opening {app_name}.\n")

def search_youtube(search_query):
    youtube_search_url = f"https://www.youtube.com/results?search_query={search_query}&sp=EgIQAQ%253D%253D"

    webbrowser.open(youtube_search_url)
    speak(f"Searching YouTube for {search_query}")
    conversation_text.insert(tk.END, f"Assistant: Playing the result for {search_query} on YouTube.\n")
def start_listening():
    global listening
    listening = True
    threading.Thread(target=listen_loop, daemon=True).start()

def stop_listening():
    global listening
    listening = False

def listen_loop():
    while listening:
        query, audio_file_path = listen()
        if query:
            conversation_text.insert(tk.END, f"You: {query}\n")
            process_query(query, audio_file_path)

def on_manual_input():
    query = manual_entry.get()
    if query:
        conversation_text.insert(tk.END, f"You: {query}\n")
        process_query(query)

def retrain_model(speaker_name):
    try:
        # Path where the speaker's audio samples are stored
        save_path = os.path.join(model_path, speaker_name)
        
        # Collect all the .wav files for the speaker
        wav_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.wav')]
        
        # Initialize an empty list to hold all the feature vectors
        features = np.array([])

        # Extract features from each .wav file
        for wav_file in wav_files:
            sr, audio = read(wav_file)
            mfcc_features = extract_features(audio, sr)
            if features.size == 0:
                features = mfcc_features
            else:
                features = np.vstack((features, mfcc_features))

        # Train the GMM model
        n_components = 16  # Number of Gaussian components
        gmm = GaussianMixture(n_components=n_components, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(features)

        # Save the trained GMM model as a .gmm file
        model_file_path = os.path.join(model_path, f"{speaker_name}.gmm")
        with open(model_file_path, 'wb') as model_file:
            pickle.dump(gmm, model_file)

        print(f"Model trained and saved for {speaker_name} at {model_file_path}")
        speak(f"Model for {speaker_name} has been retrained and saved.")
        conversation_text.insert(tk.END, f"Assistant: Model for {speaker_name} has been retrained and saved.\n")

    except Exception as e:
        print(f"Error retraining model for {speaker_name}: {e}")
        conversation_text.insert(tk.END, f"Assistant: Error retraining model for {speaker_name}.\n")


    


def record_samples(speaker_name, num_samples, duration):
    sample_rate = 16000  # Sample rate in Hz
    save_path = os.path.join(model_path, speaker_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    speak(f"Recording {num_samples} samples for {speaker_name}. Please speak after the beep.")

    for i in range(num_samples):
        speak(f"Recording sample {i + 1}")
        print(f"Recording sample {i + 1} of {num_samples}")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        file_path = os.path.join(save_path, f"{speaker_name}_{i + 1}.wav")
        sf.write(file_path, recording, sample_rate)
        print(f"Saved {file_path}")
        speak("Sample saved.")

    speak(f"All samples for {speaker_name} have been recorded.")
    conversation_text.insert(tk.END, f"Assistant: Recorded {num_samples} samples for {speaker_name}.\n")
    speak("Retraining model.")
    retrain_model(speaker_name)

def start_recording():
    speaker_name = speaker_name_entry.get()
    num_samples = int(num_samples_entry.get())
    duration = int(sample_duration_entry.get())
    record_samples(speaker_name, num_samples, duration)
    
    # After recording, retrain the model
    
    
    recording_window.destroy()

def open_recording_window():
    global recording_window, speaker_name_entry, num_samples_entry, sample_duration_entry
    recording_window = tk.Toplevel(app)
    recording_window.title("Add New Voice Samples")
    
    recording_canvas = tk.Canvas(recording_window, width=500, height=300)
    recording_canvas.pack(fill="both", expand=True)
    
    speaker_name_label = tk.Label(recording_window, text="New Speaker Name:", font=("Helvetica", 12))
    recording_canvas.create_window(150, 50, window=speaker_name_label)
    
    speaker_name_entry = tk.Entry(recording_window, width=20, font=("Helvetica", 12))
    recording_canvas.create_window(320, 50, window=speaker_name_entry)

    num_samples_label = tk.Label(recording_window, text="Number of Samples:", font=("Helvetica", 12))
    recording_canvas.create_window(150, 100, window=num_samples_label)
    
    num_samples_entry = tk.Entry(recording_window, width=20, font=("Helvetica", 12))
    recording_canvas.create_window(320, 100, window=num_samples_entry)

    sample_duration_label = tk.Label(recording_window, text="Sample Duration (s):", font=("Helvetica", 12))
    recording_canvas.create_window(150, 150, window=sample_duration_label)
    
    sample_duration_entry = tk.Entry(recording_window, width=20, font=("Helvetica", 12))
    recording_canvas.create_window(320, 150, window=sample_duration_entry)

    record_button = tk.Button(recording_window, text="Start Recording", command=start_recording, font=("Helvetica", 14), bg="orange", fg="white")
    recording_canvas.create_window(200, 200, window=record_button)

    retrain_button = tk.Button(recording_window, text="Retrain Model", command=retrain_model, font=("Helvetica", 14), bg="purple", fg="white")
    recording_canvas.create_window(300, 200, window=retrain_button)


# GUI Application
def create_app():
    global conversation_text, manual_entry, app
    app = tk.Tk()
    app.title("Voice Assistant")

    # Load and set the background image
    bg_image = Image.open(r'C:\Users\nitis\OneDrive\Desktop\img.jpg')
    bg_image = bg_image.resize((800, 600))
    bg_photo = ImageTk.PhotoImage(bg_image)

    canvas = tk.Canvas(app, width=800, height=600)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_photo, anchor="nw")

    # Add widgets on top of the background
    title_label = tk.Label(app, text="Voice Assistant", font=("Helvetica", 16), bg="white")
    canvas.create_window(400, 50, window=title_label)

    conversation_text = scrolledtext.ScrolledText(app, wrap=tk.WORD, width=50, height=15, font=("Helvetica", 12))
    canvas.create_window(400, 250, window=conversation_text)

    manual_entry = tk.Entry(app, width=50, font=("Helvetica", 12))
    canvas.create_window(400, 400, window=manual_entry)

    manual_button = tk.Button(app, text="Submit", command=on_manual_input, font=("Helvetica", 14), bg="blue", fg="white")
    canvas.create_window(400, 450, window=manual_button)

    start_button = tk.Button(app, text="Start Listening", command=start_listening, font=("Helvetica", 14), bg="green", fg="white")
    canvas.create_window(300, 500, window=start_button)

    stop_button = tk.Button(app, text="Stop Listening", command=stop_listening, font=("Helvetica", 14), bg="red", fg="white")
    canvas.create_window(500, 500, window=stop_button)

    # Button to open the recording window
    open_recording_window_button = tk.Button(app, text="Add Voice Samples", command=open_recording_window, font=("Helvetica", 14), bg="orange", fg="white")
    canvas.create_window(400, 550, window=open_recording_window_button)

    app.mainloop()


if __name__ == "__main__":
    create_app() 