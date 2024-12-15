'''
By Harry Gray
First time using MFFC?, it was for me!
https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split

'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import tkinter as tk
from tkinter import messagebox
import threading

from sklearn.model_selection import train_test_split
import librosa
from torch.utils.data import Dataset, DataLoader

 ## model_undertraining = False #checks if the model is already being trained (prevents spamming the button) ##NOT REQUIRED Thread is allready doing this

#MFFC from files (extract 13 features)
def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)  #average's MFFC across the frames! 

#dataset
class AudioDataset(Dataset): #OOP
    def __init__(self, a_files, labels, transform=None): 
        self.a_files = a_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.a_files)

    def __getitem__(self, idx):
        audio_path = self.a_files[idx]
        label = self.labels[idx]
        features = extract_mfcc(audio_path)
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

#loading files variables 
audio_dir = 'AudioFiles' #change the data set location if you like, make sure its in the same structure however.
a_files = []
labels = []

map = {'KICK': 0, 'HAT': 1, 'SNARE': 2};

#loop through each class and collect the files
for label in map.keys():
    for file in os.listdir(os.path.join(audio_dir, label)):
        if file.endswith(".wav"):
            a_files.append(os.path.join(audio_dir, label, file))
            labels.append(map[label])

#data into training and testing
X_train, X_val, y_train, y_val = train_test_split(a_files, labels, test_size=0.2, stratify=labels);

# Create datasets
train_set = AudioDataset(X_train, y_train)
val_set = AudioDataset(X_val, y_val)

#audio features and their labels
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader =  DataLoader(val_set, batch_size=32, shuffle=False)

# 13 inputs => 64 hidden layers => KICK, HAT, SNARE aka 3 Outputs
class AudioClassifier(nn.Module): #OOP
    def __init__(self):
            super(AudioClassifier, self).__init__() #call parent OPP :)
            self.fc1 = nn.Linear(13, 64) #13 MFCC features 13 => 64
            self.fc2 = nn.Linear(64, 3) #3 classes out 64 => 3

    def forward(self, x): #move through the neural network hidden layers
            x = torch.relu(self.fc1(x)) #relu improvement
            x = self.fc2(x)
            return x;

#setting up model + extra
model = AudioClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate a little smaller for extra updates

epochs_num = 500  #number of epochs before stopping learning

def train_model(): #train's model with error handling for fun
    try:
        for epoch in range(epochs_num):
            try:
                model.train()
                run_loss = 0.0
                preds_correct = 0
                preds_sum = 0

                for inputs, labels in train_loader:
                    try:
                        optimizer.zero_grad()
                        outputs = model(inputs)  # forward pass
                        loss = criterion(outputs, labels)  # calculate loss
                        loss.backward()
                        optimizer.step()  # backpropagation and optimization step

                        run_loss += loss.item()  # accumulate loss

                        _, predicted = torch.max(outputs, 1)  # max
                        preds_correct += (predicted == labels).sum().item()
                        preds_sum += labels.size(0)

                    except Exception as e:
                        print(f"Error during training batch: {e}")
                        continue  # skip this batch if an error occurs

                # print the training accuracy
                train_acc = preds_correct / preds_sum if preds_sum != 0 else 0
                print(f"Epoch {epoch+1}/{epochs_num}, Loss: {run_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")

                # validate model
                model.eval()
                preds_correct = 0
                preds_sum = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        try:
                            outputs = model(inputs)  # get the outputs
                            _, predicted = torch.max(outputs, 1)  # get the class
                            preds_correct += (predicted == labels).sum().item()  # count predictions
                            preds_sum += labels.size(0)  # total predictions

                        except Exception as e:
                            print(f"Error during validation batch: {e}")
                            continue  # skip this batch if an error occurs

                # print the accuracy
                val_accuracy = preds_correct / preds_sum if preds_sum != 0 else 0
                print(f"Validation Accuracy: {val_accuracy:.4f}")

            except Exception as e:
                print(f"Error during epoch {epoch+1}: {e}")
                continue  # skip this epoch if an error occurs

        # save the trained model so you don't have to train it again
        try:
            torch.save(model.state_dict(), 'audio_classifier.pth')
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving the model: {e}")

    except Exception as e:
        print(f"Error in training process: {e}")

#threading
train_thread = threading.Thread(target=train_model)


#--------------------------------GUI--------------------------------------
root = tk.Tk()
root.title("Train & Class")
root.configure(bg="#2E2E2E")  #most likable grey

#executes Classit_Model.py for quicker access
def class_audio():  
    root.destroy()
    os.system("python Classit_Model.py")
    exit;

#buttons
train_button = tk.Button(root, text="Train", command=train_thread.start , width=15, height=2, bg="aquamarine1")
train_button.pack(pady=10)

class_button = tk.Button(root, text="Class", command=class_audio, width=15, height=2, bg="brown1")
class_button.pack(pady=10)

#class
root.iconbitmap("icon.ico") # Icon
root.geometry("200x125") # size
root.resizable(False, False) # not resizeable

#GUI loop
root.mainloop()
