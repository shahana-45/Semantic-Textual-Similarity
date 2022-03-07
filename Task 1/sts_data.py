import pandas as pd
import numpy as np
from preprocess import Preprocess
import logging
import torch
from dataset import STSDataset
from datasets import load_dataset
import torchtext
from torchtext.legacy.data import Field
import spacy

from torch import nn
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.INFO)

"""
For loading STS data loading and preprocessing
"""


class STSData:
    def __init__(
        self,
        dataset_name,
        columns_mapping,
        stopwords_path="stopwords-en.txt",
        model_name="lstm",
        max_sequence_len=512,
        normalization_const=5.0,
        normalize_labels=False,
    ):
        # for spacy based tokenisation
        self.nlp = spacy.load("en_core_web_sm")
        """
        Loads data into memory and create vocabulary from text field.
        """
        self.normalization_const = normalization_const
        self.normalize_labels = normalize_labels
        self.model_name = model_name
        self.max_sequence_len = max_sequence_len
        self.dataset_name = dataset_name
        ## load data file into memory
        self.load_data(dataset_name, columns_mapping, stopwords_path)
        self.columns_mapping = columns_mapping
        ## create vocabulary over entire dataset before train/test split
        self.create_vocab()

    def load_data(self, dataset_name, columns_mapping, stopwords_path):
        """
        Reads data set file from disk to memory using pandas
        """
        logging.info("loading and preprocessing data...")

        ## load datasets
        self.train, self.valid, self.test = load_dataset(self.dataset_name, split=['train', 'validation', 'test'])
        self.train = self.train.remove_columns(['entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset', 'id', 'label'])
        self.valid = self.valid.remove_columns(['entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset', 'id', 'label'])
        self.test = self.test.remove_columns(['entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset', 'id', 'label'])
        
        self.train = pd.DataFrame(self.train)
        self.test = pd.DataFrame(self.test)
        self.valid = pd.DataFrame(self.valid)
        
        ## perform text preprocessing
        p = Preprocess("abc.txt")
        self.train = p.perform_preprocessing(self.train, columns_mapping)
        self.test = p.perform_preprocessing(self.test, columns_mapping)
        self.valid = p.perform_preprocessing(self.valid, columns_mapping)
       
        logging.info("reading and preprocessing data completed...")

    def create_vocab(self):
        """
        Creates vocabulary over entire text data field.
        """
        logging.info("creating vocabulary...")

        # create vocabulary
        text_field = Field(
            tokenize='basic_english',
            lower=True
        )
        label_field = Field(sequential=False, use_vocab=False)
        
        # apply preprocessing
        concat = self.train["sentence_A"] + " " + self.train["sentence_B"]
        preprocessed_text = concat.apply(lambda x: text_field.preprocess(x))
        
        # load fastext simple embedding with 300 dimensions
        text_field.build_vocab(
            preprocessed_text,
            vectors='fasttext.simple.300d'
        )
        # save the vocab object
        self.vocab = text_field.vocab        
        
        logging.info("creating vocabulary completed...")

    def data2tensors(self, data):
        """
        Converts raw data sequences into vectorized sequences as tensors
        """
        # create embedding vectors (using the vocab) from word tokens
        data['sentence_A'] = data['sentence_A'].apply(lambda x: self.vectorize_sequence(x))
        data['sentence_B'] = data['sentence_B'].apply(lambda x: self.vectorize_sequence(x))
        # normalise labels
        data['relatedness_score'] = (data['relatedness_score'] / self.normalization_const)
        
        # create list of tensors
        sen1 = []
        sen2 = []
        
        for index, row in data.iterrows():
            sen1.append(torch.LongTensor(row['sentence_A']))
            sen2.append(torch.LongTensor(row['sentence_B']))
            
        return (sen1, sen2, torch.FloatTensor(data['relatedness_score']))
              
            
    def get_data_loader(self, batch_size=8):
        data_loaders = dict()
        
        train_data = self.train        
        # convert to list of tensors
        sen1, sen2, labels = self.data2tensors(train_data)
        
        lengths_1 = [len(row) for row in sen1]
        lengths_2 = [len(row) for row in sen2]
        
        # pad to maximum length of sentence in dataset
        sen1 = self.pad_sequences(sen1, torch.tensor(lengths_1))
        sen2 = self.pad_sequences(sen2, torch.tensor(lengths_2))
                
        # create the training dataset and loader
        train_dataset = STSDataset(sen1, sen2, labels, torch.tensor(lengths_1), torch.tensor(lengths_2))       
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        data_loaders["train"] = train_loader

        # create valid loader
        valid_data = self.valid        
        # convert to list of tensors
        sen1, sen2, labels = self.data2tensors(valid_data)
        
        lengths_1 = [len(row) for row in sen1]
        lengths_2 = [len(row) for row in sen2]
        
        # pad to maximum length of sentence in dataset
        sen1 = self.pad_sequences(sen1, torch.tensor(lengths_1))
        sen2 = self.pad_sequences(sen2, torch.tensor(lengths_2))
                
        # create the validation dataset and loader
        valid_dataset = STSDataset(sen1, sen2, labels, torch.tensor(lengths_1), torch.tensor(lengths_2))       
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)
        data_loaders["valid"] = valid_loader    
        
        # create test loader
        test_data = self.test        
        # convert to list of tensors
        sen1, sen2, labels = self.data2tensors(test_data)
        
        # calculate length of each sentence
        lengths_1 = [len(row) for row in sen1]
        lengths_2 = [len(row) for row in sen2]
        
        # pad to maximum length of sentence in dataset
        sen1 = self.pad_sequences(sen1, torch.tensor(lengths_1))
        sen2 = self.pad_sequences(sen2, torch.tensor(lengths_2))
        
        # create the test dataset and loader
        test_dataset = STSDataset(sen1, sen2, labels, torch.tensor(lengths_1), torch.tensor(lengths_2))       
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
        data_loaders["test"] = test_loader         
        
        return data_loaders 

    def sort_batch(self, batch, targets, lengths):
        """
        Sorts the data, lengths and target tensors based on the lengths
        of the sequences from longest to shortest in batch
        """
        sents1_lengths, perm_idx = lengths.sort(0, descending=True)
        sequence_tensor = batch[perm_idx]
        target_tensor = targets[perm_idx]
        return sequence_tensor.transpose(0, 1), target_tensor, sents1_lengths

    def vectorize_sequence(self, sentence):
        """
        Replaces tokens with their indices in vocabulary
        """
        # tokenise using spacy and then vectorise
        indices = []
        tokens = self.nlp(sentence)
        for token in tokens:
            indices.append(self.vocab[token.text])
        return indices

    def pad_sequences(self, vectorized_sents, sents_lengths):
        """
        Pads zeros at the end of each sequence in data tensor till max
        length of sequence in that batch
        """
        x_padded = pad_sequence(vectorized_sents, padding_value=0, batch_first=True)
        return x_padded