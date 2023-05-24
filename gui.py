import os
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import re
import gradio as gr
from transcription import *

def run_gui():
    gr.Interface(
    title = 'Whisper with Speaker Recognition',
    fn=transcribe, 
    inputs=[
        gr.inputs.Audio(source="upload", type="filepath"),
        gr.inputs.Number(default=2, label="Number of Speakers")

    ],
    outputs=[
        gr.outputs.Textbox(label='Transcript')
    ]
  ).launch()

if __name__ == "__main__":
    run_gui()