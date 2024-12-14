Hello,

Due to some of the samples used, which are not fair use, this git repo is private.

Libarys Needed:
torch #For the network
librosa #A libary for all things audio data analysis
numpy #for the tensors and arrays etc
tkinter # GUI
matplotlib #Graph's

Quick Desc of Script's & Dir's:

AudioFiles/KICK #Audio files for kicks
AudioFiles/HAT #Audio files for hats
AudioFiles/SNARE #Audio files for snares
.conda # conda libs of course.

Train_Model.py -
This reads from the AudioFiles/* then takes 13 features (MFFC) from each audio file, the directory is how the class is identifyed.
Create's a model 13 inputs 2 x 64 hidden layer 3 outputs KICK HAT SNARE. (This is to predict the class of the selected wav file).
Trains the model, and dumps it as a dict, .pth. (This helps if you dont want to train the model yourself) (it takes less then a minute) (highly depends on how big i make my dataset).

Classit.py -
This a simple gui, that takes the model dict and use's the trained model to predict the class of the sample. (with a higher dataset this gets quite good at predicting).
Uses GUI and add's a graph for extra clarification.




