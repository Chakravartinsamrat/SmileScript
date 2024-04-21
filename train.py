import json
import numpy as np
from nltk_utils import tokenize, lemmatize, bag_of_words  # Change stem to lemmatize

import torch
import torch.nn as nn
from chat_dataset import ChatDataset
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from nltk.stem import WordNetLemmatizer  # Import WordNetLemmatizer

if __name__ == '__main__':
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []
    ignore_letters = ['?', '!']
    conversational_history = []
    
    lemmatizer = WordNetLemmatizer()  # Initialize the lemmatizer
    
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            # Change stemming to lemmatization here
            w = [lemmatizer.lemmatize(word) for word in w]  # Lemmatize each word
            all_words.extend(w)
            xy.append((w, tag))

    all_words = [lemmatizer.lemmatize(w) for w in all_words if w not in ignore_letters]  # Lemmatize again to ensure consistency
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Train model
    X_train = []
    Y_train = []
    #initialized 2 empty list 
    for (pattern_sentence, tag) in xy:
        #iterates over patterns and tag eg: [{what}{are}{you}, {greeting}]
        bag = bag_of_words(pattern_sentence, all_words)
        #bag words function is called
        X_train.append(bag)
        #appends bag to xtrain
        label = tags.index(tag)
        #appends tags to y train
        Y_train.append(label)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    #covert to numpy array


    # Hyperparameters
    batch_size = 8 #number of samples per batch duriong trianing
    hidden_size = 16 #numbe rof units in hidden layer
    
    output_size = len(tags)  #number of outputs, detrmined by tags
    input_size = len(X_train[0])        #size of input, determined by bag of words
    learning_rate = 0.001               #learning rate for optimization algorithm
    num_epochs = 1000                   #number of times the entire dataset is passed forward and backward through the neural network
#epochs= one complete pass through the entire training dataset, 
    dataset = ChatDataset(X_train, Y_train)   #load dataset
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)  
    #creates a dataloader object to handle batching of the training data , shuffles the data for better generalization 

    
 #defines devices used for
    device = torch.device('cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    #initializes NN with the given parameters

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    #defines loss function, it calculates the difference between predicted porbabbility distribution and actual probability distribution
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #adaptive moment distribution 

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            #within each epoch, code iterates over batches of data obtained from trainloader
            words = words.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            #datatype to longtensor??????
            #moving data to device
            
            # Forward pass
            outputs = model(words)
            #passes words through THE NN
            loss = criterion(outputs, labels)
            #calculates loss between predicted  "outputs " and actual "Labels" 

            # Backward and optimize
            optimizer.zero_grad()
            #cleaars gradient of all optimized parameters before backpass, to ensure prev gradients dont get collected
            loss.backward()
            #computes gradient loss
            optimizer.step()
            #updates amodel parameters using computed gradients and optimization algo

#logging progress
        if (epoch + 1) % 100 == 0: #to check if epoch is a multiple of 100
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final Loss: {loss.item():.4f}')


#Now the trainning data is saved to Data.pth, IT IS ALSO CALLED A DICTIONARY
data={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size": output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags,


}

FILE ="data.pth"
torch.save(data,FILE)

print(f'training Complete. File Saved to {FILE}')

 