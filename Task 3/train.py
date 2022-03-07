import torch
from torch import nn
import logging
from tqdm import tqdm
from torch.autograd import Variable
import sklearn
from sklearn.metrics import mean_squared_error
import numpy as np

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
        
        total_loss = []
        running_loss = 0.0
        n_batches = 0
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
                y_pred = model(input1, input2, sen1_lengths, sen2_lengths)

                loss = criterion(y_pred, labels)
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
                model, data, criterion, dataloader["valid"], config_dict, device
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

        y_pred = model(input1, input2, sen1_lengths, sen2_lengths)

        loss = criterion(y_pred, labels)
        
        running_vloss += loss.item()
       
        acc = mean_squared_error(labels.detach().numpy(), y_pred.detach().numpy())
        total_accuracy += acc
        v_loss = running_vloss
        print('Validation loss: %.3f' % (running_vloss))
        print('Validation set accuracy: %.3f' % (1 - acc))
        running_vloss = 0.0
               
    return ((1-(total_accuracy/n_batches)), v_loss)
