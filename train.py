import torch
from torch import nn
import logging
from tqdm import tqdm
from torch.autograd import Variable


logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, dataloader, data, max_epochs, config_dict):
    device = config_dict["device"]
    criterion = nn.MSELoss()
    max_accuracy = 5e-1
    
    for epoch in tqdm(range(max_epochs)):
        # TODO implement
        model.train()
        
        print("Running EPOCH",epoch+1)
        total_loss = 0.0
        n_batches = 0
        correct = 0
        # dataloader contains training data batches
        for batch_idx,train in enumerate(dataloader["train"]):
            
            sen1_batch, sen2_batch, sen1_lengths, sen2_lengths, labels = Variable(train[0]), Variable(train[1]), train[2], train[3], train[4]
            
            #print(train[4])
            optimizer.zero_grad()
            
            # calculate similarity score prediction
            y_pred, A1, A2  = model(sen1_batch, sen2_batch, sen1_lengths, sen2_lengths)
            
            loss = criterion(y_pred.float(), labels.float())
            #print(attention_penalty_loss(A2, 0.0, device))
                      
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            #print(labels.size()[0])
            if batch_idx % 20 == 19:    # print every 20 batches
                print('Running loss: ', (total_loss / 20))
                total_loss = 0.0
                
        
        # TODO: computing accuracy using sklearn's function
        # acc = 

        ## compute model metrics on dev set
        model.eval()
        print("Evaluating validation set ....")
        val_acc = evaluate_dev_set(
            model, data, criterion, dataloader["valid"], config_dict, device
        )

        if val_acc > max_accuracy:
            max_accuracy = val_acc
            logging.info(
                "new model saved"
            )  ## save the model if it is better than the prior best
            torch.save(model.state_dict(), "{}.pth".format(config_dict["model_name"]))
            

        '''
        logging.info(
            "Train loss: {} - acc: {} -- Validation loss: {} - acc: {}".format(
                torch.mean(total_loss.data.float()), acc, val_loss, val_acc
            )
        )
        '''
        
    print('Finished Training')
    return model


def evaluate_dev_set(model, data, criterion, data_loader, config_dict, device):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on dev set")
    running_vloss = 0.0
    
    for k, vdata in enumerate(data_loader):
        sen1_batch, sen2_batch, sen1_lengths, sen2_lengths, labels = vdata
        target, A1, A2 = model(sen1_batch, sen2_batch, sen1_lengths, sen2_lengths)
        
        loss = criterion(target, labels)
        running_vloss += loss.item()
        

        print('Validation loss: %.3f' % (running_vloss))
        running_vloss = 0.0
            
    return 0

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
    # implement
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
    # implement
    return torch.sum(torch.sum(torch.sum(annotation_mul_difference**2,1),1)**0.5).type(torch.FloatTensor)