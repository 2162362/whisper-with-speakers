import os
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import re
import gradio as gr
from transcription import transcribe

# Function to open file dialog
def open_file_dialog():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav;*.m4a"), ("Video Files", "*.mp4;*.avi")])

# Function to transcribe audio
def gui_transcribe():
    audio = file_path
    num_speakers = float(num_speakers_entry.get())
    transcript = transcribe(audio, num_speakers)
    print(transcript)

def run_gui():
    # Create the main window
    window = tk.Tk()
    window.title("Whisper with Speaker Recognition")

    # Create an Audio File Input Button
    audio_input_button = tk.Button(window, text="Upload Audio", command=open_file_dialog)
    audio_input_button.pack(pady=10)

    # Create a Number of Speakers Input Entry
    num_speakers_label = tk.Label(window, text="Number of Speakers:")
    num_speakers_label.pack()

    global num_speakers_entry
    num_speakers_entry = tk.Entry(window, width=2)
    num_speakers_entry.insert(tk.END, "2")  # Set default value to 2
    num_speakers_entry.pack(pady=10)

    # Create a Transcribe Button
    transcribe_button = tk.Button(window, text="Transcribe", command=gui_transcribe)
    transcribe_button.pack(pady=10)

    # Start the main loop
    window.mainloop()

if __name__ == "__main__":
    run_gui()