import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import similarity_score
import copy

"""
Wrapper class using Pytorch nn.Module to create the architecture for our model
Architecture is based on the paper: 
A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
https://arxiv.org/pdf/1703.03130.pdf
"""


class SiameseBiLSTMAttention(nn.Module):
    def __init__(
        self,
        batch_size,
        output_size,
        hidden_size,
        vocab_size,
        embedding_size,
        embedding_weights,
        lstm_layers,
        device,
        bidirectional,
        self_attention_config,
        fc_hidden_size,
    ):
        super(SiameseBiLSTMAttention, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lstm_layers = lstm_layers
        self.device = device
        self.bidirectional = bidirectional
        self.fc_hidden_size = fc_hidden_size
        self.lstm_directions = (
            2 if self.bidirectional else 1
        )  ## decide directions based on input flag
                       
        ## model layers
        # initialize the look-up table.
        print(self.embedding_size)
        embedding = nn.Embedding(vocab_size, self.embedding_size)
        
        # assign the look-up table to the pre-trained fasttext word embeddings.
        self.embeddings = torch.nn.Embedding.from_pretrained(embedding_weights)
        #print(self.embeddings)
        
        ## initialize lstm layer
        self.bilstm = nn.LSTM(self.embedding_size, self.lstm_hidden_size, self.lstm_layers, bidirectional=self.bidirectional)
        #self.hidden_state = self.init_hidden(self.batch_size)
        ## initialize self attention layers
        
        ## incase we are using bi-directional lstm we'd have to take care of bi-directional outputs in
        ## subsequent layers
        self.W_s1 = nn.Linear(self.lstm_directions*self.lstm_hidden_size, self_attention_config["hidden_size"])
        self.W_s1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.W_s1.weight)
        self.W_s2 = nn.Linear(self_attention_config["hidden_size"], self_attention_config["output_size"])
        torch.nn.init.xavier_uniform(self.W_s2.weight)
        self.W_s2.bias.data.fill_(0)
        
        self.hidden_state = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """

        return (torch.zeros(self.lstm_directions*self.lstm_layers,self.batch_size,self.lstm_hidden_size, requires_grad=True),
                torch.zeros(self.lstm_directions*self.lstm_layers,self.batch_size,self.lstm_hidden_size, requires_grad=True))
        

    def forward_once(self, batch, lengths):
        """
        Performs the forward pass for each batch
        """
        
        ## batch shape: (num_sequences, batch_size)
        ## embeddings shape: (seq_len, batch_size, embedding_size)
        
        # TODO implement
        # pass batch of sentences and retrieve embeddings from it (now we have 3d tensor)
        #print(batch.size())
        inputs2 = batch.permute(1, 0, 2)
        
        outputs, (hn, cn) = self.bilstm(inputs2,self.hidden_state)    
        #print(outputs.size())

        outputs2 = torch.permute(outputs, (1, 0, 2))
        
        # calculate self_attention
        annotation_matrix = self.W_s2(torch.tanh(self.W_s1(outputs2)))
        annotation_matrix1 = annotation_matrix.permute(0, 2, 1)
        annotation_matrix2 = F.softmax(annotation_matrix1, dim=2)
        
        #print(annotation_matrix)
         
        # generate self-attentive sentence embeddings
        sentence_embeddings = annotation_matrix2@outputs2   # M = AH matrix from ICLR paper
        
        #print(sentence_embeddings)
        return sentence_embeddings, annotation_matrix2
        
        #print(sentence_embeddings.size())
        #return sentence_embeddings, annotation_matrix2
        
        
    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths):
        """
        Performs the forward pass for each batch
        """
        ## init context and hidden weights for lstm cell
        self.hidden_state = self.init_hidden(self.batch_size)
        
        # implement forward pass on both sentences. calculate similarity using similarity_score()
        output1, A1 = self.forward_once(sent1_batch, sent1_lengths)
        output2, A2 = self.forward_once(sent2_batch, sent2_lengths)
        #print(output1.size())
        #print(output2)
        # TODO check how to pass data to similarity score
        score = similarity_score(output1, output2)

        return score, A1, A2

    
# NOTE: This class is not used for now
class SelfAttention(nn.Module):
    """
    Implementation of the attention block
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        # TODO implement
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
               

    ## the forward function would receive lstm's all hidden states as input
    def forward(self, attention_input):
        # TODO implement
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(attention_input)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix
        