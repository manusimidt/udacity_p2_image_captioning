import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size;
        self.hidden_size = hidden_size;
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size)) 
        self.vocab_size = vocab_size;
        self.num_layers = num_layers;
        
        self.emb = nn.Embedding(vocab_size, embed_size)
        
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # connecting layer
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        # todo what shall i do with <start> and <end>
        embeddings = self.emb(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        output, hidden = self.lstm(embeddings)
        self.hidden = hidden
        return self.fc1(output)
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        
        # initialize the hidden state
        hidden_states = states
        
        for _ in range(max_len):
            # pass the input through the lstm layers
            lstm_out, hidden_states = self.lstm(inputs, hidden_states)
            # print(lstm_out)
            # pass the output of the lstm layer through the fully connected layer
            outputs = self.fc1(lstm_out) 
            outputs = outputs.squeeze(1)   
            # print(f"max idx: {output.max(1)[1][0]}")
            wordid  = outputs.argmax(dim=1)
            caption.append(wordid.item())
            inputs = self.emb(wordid.unsqueeze(0))
        return caption
            
        
    