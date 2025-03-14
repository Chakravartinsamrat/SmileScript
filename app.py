from flask import Flask, render_template, request
import random
import json
import torch
import nltk
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__,static_folder='static')

device = torch.device('cpu')
# nltk.download('punkt')
# Load intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load pretrained model
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Load neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "SmileScript"

def get_response(msg):
    sentence = msg
    if sentence == "quit":
        return None
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.20:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return f"{bot_name}: {random.choice(intent['responses'])}"
    else:
        return f"{bot_name}: I do not Understand..."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_msg = request.args.get('msg')
    return get_response(user_msg)

@app.route("/debug")
def debug():
    return "Debug route is working"

if __name__ == "__main__":
    app.run(debug=True)
