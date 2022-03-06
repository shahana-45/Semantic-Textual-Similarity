import torch
from torch import nn
import logging
from tqdm import tqdm
from torch.autograd import Variable
import sklearn
from sklearn.metrics import mean_squared_error
import numpy as np
import Trans_Encoder
#from Trans_Encoder import Embedder,PositionalEncoder, MultiHeadAttention, FeedForward, Norm, EncoderLayer, Encoder, Transformer
from importlib import reload
reload(Trans_Encoder)

logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, dataloader, data, max_epochs, config_dict):
    device = config_dict["device"]
    acc = 0.0
    max_accuracy = 5e-1
    criterion = torch.nn.MSELoss()
    n_batches = 0
    
    for epoch in tqdm(range(max_epochs)):
        # TODO implement
        model.train()
        
        print("Running EPOCH",epoch+1)

        #hidden = model.init_hidden(model.batch_size)
        
        total_loss = []
        running_loss = 0.0
        n_batches = 0
        #correct = 0
        acc = 0.0
        total_train_acc = 0.0
        
        with torch.autograd.set_detect_anomaly(True):
        # dataloader contains training data batches
            for batch_idx,train in enumerate(dataloader["train"]):
                n_batches += 1
                sen1_batch, sen2_batch, sen1_lengths, sen2_lengths, labels = train[0], train[1], train[2], train[3], train[4]
                                 
                input1 = nn.Parameter(model.embeddings(sen1_batch).float())
                input2 = nn.Parameter(model.embeddings(sen2_batch).float()) 
                                                
                input1.requires_grad_(True)
                input2.requires_grad_(True)
                
                optimizer.zero_grad()

                # calculate predicted similarity score 
                y_pred, A1, A2, attn_scores1, attn_scores2 = model(input1, input2, sen1_lengths, sen2_lengths)

                # calculate penalisation term
                penalty1 = attention_penalty_loss(A1, config_dict["self_attention_config"]["penalty"], device)
                penalty2 = attention_penalty_loss(A2, config_dict["self_attention_config"]["penalty"], device)

                loss = custom_loss(y_pred.float(), labels.float(), penalty1, penalty2, model.batch_size)
                
                loss.backward()
                
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                
                optimizer.step()
                               
                total_loss.append(loss.item())
                running_loss += loss.item()

                # compute accuracy using sklearn's function
                mse = mean_squared_error(labels.detach().numpy(), y_pred.detach().numpy())
                acc += mse
                total_train_acc += mse

                if batch_idx % 10 == 9:    # print every 10 batches
                    print('Running loss: ', (running_loss / 10))
                    running_loss = 0.0
                    print('Training set accuracy:', (1 - (acc / 10)))
                    acc = 0.0
                   
                
            ## compute model metrics on dev set
            model.eval()
            print("Evaluating validation set ....")
            val_acc, val_loss = evaluate_dev_set(
                model, data, loss, dataloader["valid"], config_dict, device
            )

            # if greater than 50% accuracy on validation set, save the model
            if val_acc > max_accuracy:
                max_accuracy = val_acc
                logging.info(
                    "new model saved"
                )  ## save the model if it is better than the prior best
                torch.save(model.state_dict(), "{}.pth".format(config_dict["model_name"]))

            total_train_acc = 1 - (total_train_acc/n_batches)
            
            logging.info(
                "Train loss: {} - acc: {} -- Validation loss: {} - acc: {}".format(
                    torch.mean(torch.Tensor(total_loss)), total_train_acc, val_loss, val_acc
                )
            )
            
            n_batches = 0
        
           
        print('Finished Training')
    return model


def evaluate_dev_set(model, data, criterion, data_loader, config_dict, device):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on dev set")
    running_vloss = 0.0
    v_loss = 0.0
    acc = 0.0
    n_batches = 0
    total_accuracy = 0.0
    
    for k, vdata in enumerate(data_loader):
        n_batches += 1
        sen1_batch, sen2_batch, sen1_lengths, sen2_lengths, labels = vdata
        
        input1 = nn.Parameter(model.embeddings(sen1_batch).float())
        input2 = nn.Parameter(model.embeddings(sen2_batch).float())
                
        #input1.requires_grad_(True)
        #input2.requires_grad_(True)
        y_pred, A1, A2, attn_scores1, attn_scores2 = model(input1, input2, sen1_lengths, sen2_lengths)
        #p = y_pred[:, 1]
        penalty1 = attention_penalty_loss(A1, config_dict["self_attention_config"]["penalty"], device)
        penalty2 = attention_penalty_loss(A2, config_dict["self_attention_config"]["penalty"], device) 
                      
        loss = custom_loss(y_pred.float(), labels.float(), penalty1, penalty2, model.batch_size)
        running_vloss += loss.item()
       
        acc = mean_squared_error(labels.detach().numpy(), y_pred.detach().numpy())
        total_accuracy += acc
        v_loss = running_vloss
        print('Validation loss: %.3f' % (running_vloss))
        print('Validation set accuracy: %.3f' % (1 - acc))
        running_vloss = 0.0
               
    return ((1-(total_accuracy/n_batches)), v_loss)


def custom_loss(output, target, penalty1, penalty2, batch_size):
        # computes total loss = mse + penalty (sentence A) + penalty (sentence B)
        inter_loss = torch.mean((output - target)**2)
        loss = inter_loss + ((penalty1 + penalty2) / batch_size)
        return loss
    
def attention_penalty_loss(annotation_weight_matrix, penalty_coef, device):
    """
    This function computes the loss from annotation/attention matrix
    to reduce redundancy in annotation matrix and for attention
    to focus on different parts of the sequence corresponding to the
    penalty term 'P' in the ICLR paper
    ----------------------------------
    'annotation_weight_matrix' refers to matrix 'A' in the ICLR paper
    annotation_weight_matrix shape: (batch_size, attention_out, seq_len)
    """
    batch_size, attention_out_size = annotation_weight_matrix.size(0), annotation_weight_matrix.size(1)
    attT = annotation_weight_matrix.transpose(1,2)
    identity = torch.eye(attention_out_size)
    identity = Variable(identity.unsqueeze(0).expand(batch_size,attention_out_size,attention_out_size))
    penalty = frobenius_norm(annotation_weight_matrix@attT - identity)
    return penalty_coef*penalty


def frobenius_norm(annotation_mul_difference):
    """
    Computes the frobenius norm of the annotation_mul_difference input as matrix
    """
    return torch.sum(torch.sum(torch.sum(annotation_mul_difference**2,1),1)**0.5).type(torch.FloatTensor)