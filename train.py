import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn
from chat_dataset import ChatDataset
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

if __name__ == '__main__':
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []
    ignore_letters = [',', '.', '?', '!']

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    all_words = [stem(w) for w in all_words if w not in ignore_letters]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Train model
    X_train = []
    Y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)

        label = tags.index(tag)
        Y_train.append(label)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)


    # Hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    learning_rate = 0.001
    num_epochs = 1000

    dataset = ChatDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

#I dont fully understand how neural net works and chat funcs as crossentrophyloss and more below
    

    device = torch.device('cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final Loss: {loss.item():.4f}')
