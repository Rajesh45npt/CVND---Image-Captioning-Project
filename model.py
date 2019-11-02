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
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embeddings = nn.Embedding(self.vocab_size, embed_size)
        
        self.RNN_layers = nn.LSTM(embed_size, self.hidden_size, num_layers=num_layers)
        self.linear_layer = nn.Linear(hidden_size, self.vocab_size)
        self.activation_layer = nn.Softmax(dim=2)
        
    
    def forward(self, features, captions):
#         import pdb; pdb.set_trace()

#       Remove the last token of the end word
        captions = captions[:,:-1]
        x = self.word_embeddings(captions)
        
#       Add dummy dimension on features to match the dimension in captions
        features = features.unsqueeze(dim=1)
    
#       Add feature at the begining of the caption to pass to the lstm model
        input_tensor = torch.cat((features, x), dim=1)
        
        out, hidden = self.RNN_layers(input_tensor)
        
        y = self.linear_layer(out)
        scores = self.activation_layer(y)
        return scores

    def sample(self, inputs, states=None, max_len=20):
#         import pdb; pdb.set_trace()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        hidden_init = (torch.randn(1,1,self.hidden_size).to(device), torch.randn(1,1,self.hidden_size).to(device))
        
        """Pass the feature of the images as the input"""
        out,hidden = self.RNN_layers(inputs,hidden_init)
        
        start_token_id = 0
        end_token_id = 1
            
        
        output_lst = []
        
        output_token_id = start_token_id
        output_lst.append(output_token_id)
        for word_pos in range(max_len):
            x = torch.Tensor([output_token_id]).view(1,1).long().to(device)
            x = self.word_embeddings(x)
            out, hidden = self.RNN_layers(x, hidden)
            x = self.linear_layer(out)
            x = self.activation_layer(x)
#             print("Probabilities: ", x)
            x = torch.argmax(x)
            output_token_id = x.item()
            output_lst.append(output_token_id)
            if output_token_id == end_token_id:
                break
        else:
            output_lst.append(end_token_id)
        return output_lst