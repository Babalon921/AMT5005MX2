'''
By Harry Gray
'''
#required libarys
import torch
from torch import nn
import librosa
import numpy as np

#tkinter for gui and graphs
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#same as the one used for training
class AudioClassifier(nn.Module): #OOP
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(13, 64)#13 MFFC IN'ss
        self.fc2 = nn.Linear(64, 3)#KICK HAT SNARE 0 1 2

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#load the model
model = AudioClassifier()
model.load_state_dict(torch.load('audio_classifier.pth'))
model.eval()  # EVAULATE THE MODEL #

#process the sample file
def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)  #average over time frames

#class for a new sample
def predict_sample(audio_file_path, model):
    #extract MFCC features
    features = extract_mfcc(audio_file_path)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    #through the model
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs, 1)

    #gets the prediction to the class
    label_map = {0: 'KICK', 1: 'HAT', 2: 'SNARE'}
    predicted_label = label_map[predicted.item()]
    return predicted_label

#visual audio wave
def visualize_audio_waveform(audio_path):
    plt.clf()  # plt.clf() doesnt work fix this!!
    
    y, sr = librosa.load(audio_path, sr=None)
    plt.figure(figsize=(5, 2), facecolor='#2E2E2E')
    ax = plt.gca() 
    ax.set_facecolor('#2E2E2E') #set background of axises to background of GUI (better looking)
    plt.plot(y,color='red')
    plt.title("Waveform",color='white') #american spelling color 
    plt.xlabel("Time",color='white')
    plt.ylabel("Amplitude",color='white') 
    plt.tight_layout()
    
    #embed the plot
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.get_tk_widget().pack()
    canvas.draw()

#open file dialog + predict class and visualize waveform
def open_file_and_predict():
    file_path = filedialog.askopenfilename(title="Select a .WAV file", filetypes=[("WAV files", "*.wav")])

    if file_path:
        try:
            predicted_class = predict_sample(file_path, model)
            
            preds_labelel.config(text=f"Class: {predicted_class}")

            visualize_audio_waveform(file_path)
            
        except Exception as e:
            messagebox.showerror("error", f"An error occurred: {e}")

#GUI with a dark theme
root = tk.Tk()
root.title("Classy Finder")
#background color of the window
root.configure(bg="#2E2E2E")  #Most likable grey out there

#button to open the file dialog and predict
predict_button = tk.Button(root, text="Select A .WAV File", command=open_file_and_predict, bg="#444444", fg="white", font=("Arial", 12))
predict_button.pack(pady=20)

#dark theme labels
preds_labelel = tk.Label(root, text="Class: None", font=("Futura", 14), fg="white", bg="#2E2E2E")
preds_labelel.pack(pady=10)

root.iconbitmap("icon.ico") # Icon
root.geometry("400x300") # size
root.resizable(False, False) # not resizeable

#run mainloop
root.mainloop()