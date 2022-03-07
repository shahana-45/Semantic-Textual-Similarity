import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from utils import similarity_score
import copy


class SiameseDAN(nn.Module):
    def __init__(
        self,
        batch_size,
        vocab_size,
        embedding_size,
        embedding_weights,
        device,
    ):
        super(SiameseDAN, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.device = device
      
        ## model layers
        # initialize the look-up table.
        embedding = nn.Embedding(vocab_size, self.embedding_size)
        
        # assign the look-up table to the pre-trained fasttext word embeddings.
        self.embeddings = torch.nn.Embedding.from_pretrained(embedding_weights)
                     
        ## initialise DAN layers
        self.sizes = (300, 128, 64, 32, 16, 8)       
        self.linear_1 = nn.Linear(in_features=self.sizes[0], out_features=self.sizes[1])
        self.linear_2 = nn.Linear(in_features=self.sizes[1], out_features=self.sizes[2])
        self.linear_3 = nn.Linear(in_features=self.sizes[2], out_features=self.sizes[3])
        self.linear_4 = nn.Linear(in_features=self.sizes[3], out_features=self.sizes[4])
        self.linear_5 = nn.Linear(in_features=self.sizes[4], out_features=self.sizes[5])
               

    def forward_once(self, batch, lengths):
        """
        Performs the forward pass for each batch
        """      
        ## batch shape: (num_sequences, batch_size)
        ## embeddings shape: (seq_len, batch_size, embedding_size)

        # pass through Deep Avg. Network:        
        # compute average of input embeddings to create DAN sentence embeddings
        x = batch.mean(dim=1)
        
        # pass through linear layer(s)
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        x = F.relu(x)
        x = self.linear_3(x)
        x = F.relu(x)
        x = self.linear_4(x)
        x = F.relu(x)
        x = self.linear_5(x)        
        return x
        
        
    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths):
        """
        Performs the forward pass for each batch
        """
        
        # implement forward pass on both sentences. calculate similarity using similarity_score()
        output1 = self.forward_once(sent1_batch, sent1_lengths)
        output2 = self.forward_once(sent2_batch, sent2_lengths)
        
        score = similarity_score(output1, output2)
        #print(score)
        return score

        