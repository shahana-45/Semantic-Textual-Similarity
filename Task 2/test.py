from scipy import stats
import logging
from sklearn.metrics import mean_squared_error
import torch.nn as nn


def evaluate_test_set(model, data_loader, config_dict):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on test set")

    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on test set")
    total_acc = 0.0
    acc = 0.0
    n_batches = 0
    first_sen = 0
    
    model.eval()
    for k, vdata in enumerate(data_loader["test"]):
        n_batches += 1
        
        sen1_batch, sen2_batch, sen1_lengths, sen2_lengths, labels = vdata
        
        input1 = nn.Parameter(model.embeddings(sen1_batch).float())
        input2 = nn.Parameter(model.embeddings(sen2_batch).float())
        
        y_pred, A1, A2, attn_scores1, attn_scores2 = model(input1, input2, sen1_lengths, sen2_lengths)
        
        # return attention scores of the first sentences for heatmap visualisation
        attn_scores_1 = 0
        if first_sen == 0:
            attn_scores_1= attn_scores1[0][None, :]
            attn_scores_2= attn_scores2[0][None, :]
            first_sen = 1
        break
        
        mse = mean_squared_error(labels.detach().numpy(), y_pred.detach().numpy()) 
        acc += mse    
        total_acc += mse
        
    print('Finished testing..............')
    print('Total test set accuracy: %.3f' % (1 - (total_acc/n_batches)))

    return attn_scores_1, attn_scores_2
  
    