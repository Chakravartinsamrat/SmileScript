#data.pth is a dictionary which contains the parameters(weights and biases) of the model that were saved during training, by loding this state
#model adopts to learned parameters , allowing it to make predictions based on the learned patterns in the data

import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device= torch.device('cpu')
#SET DEVICE 
with open ('intents.json','r') as f:
    intents =json.load(f)
#LOAD FILE
    
FILE = "data.pth"
data= torch.load(FILE)
#LOAD PRETRAINED NN MODEL 
input_size= data["input_size"]
hidden_size= data["hidden_size"]
output_size= data["output_size"]
all_words= data["all_words"]
tags =data["tags"]
model_state=data["model_state"]

#loading NN
model= NeuralNet(input_size, hidden_size,output_size).to(device)
#load pretrained dictionary
model.load_state_dict(model_state)
model.eval()
#eval= evalution mode


bot_name= "SmileScript"
print("Let's Chat! type 'quit' to exit")
while True:
    sentence= input('You: ')
    if sentence == "quit":
        break
    sentence=tokenize(sentence)
    #tokenize input
    X= bag_of_words(sentence, all_words)
    X= X.reshape(1,X.shape[0])
    #reshape bagofwords vector to have a batch dimention of 1, bc the modelexpects input in batch format
    X= torch.from_numpy(X)
    #coverts into numoy array
#model prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
#extracts the index of class with highest prob from output predictions
#retrives corresponmdiong tag(intent) 

    probs= torch.softmax(output, dim=1)
    #probability func, ensure prob<=1
    prob= probs[0][predicted.item()]
    #retrives the prob corresponding to the predicted class from softmax probabilities

    if prob.item() > 0.05 :
        #if prob >0.75= confident
        for intent in intents ["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}:I do not Understand...")