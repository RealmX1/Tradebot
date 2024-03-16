import torch.optim as optim
import time
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
import torch.nn as nn
import os


import matplotlib.pyplot as plt

def train_model(model, num_epochs = 10):

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Lists to keep track of losses
    train_losses = []
    test_losses = []

    start_time = time.time()


    try:
        for epoch in range(num_epochs):
            print(epoch)
            model.train()  # Set the model to training mode
            total_train_loss = 0
            total_test_loss = 0


            total_train_tp = 0
            total_train_tn = 0
            total_train_fp = 0
            total_train_fn = 0

            total_test_tp = 0
            total_test_tn = 0
            total_test_fp = 0
            total_test_fn = 0

            for inputs, targets in train_loader:
                
                inputs = inputs.cuda()
                targets = targets.cuda()
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                # Calculate Confusion Matrix Statistics
                y_batch = targets.cpu().detach().numpy()
                y_pred  = outputs.cpu().detach().numpy()
                tp, tn, fp, fn = tfpn_calculator(y_batch, y_pred)

                total_train_tp += tp
                total_train_tn += tn
                total_train_fp += fp
                total_train_fn += fn

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            print("Training Confusion Matrix:")
            confusion_matrix_printout(total_train_tp,
                                        total_train_tn,
                                        total_train_fp,
                                        total_train_fn)

            # Evaluation phase
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_test_loss += loss.item()

                    y_batch = targets.cpu().detach().numpy()
                    y_pred  = outputs.cpu().detach().numpy()
                    tp, tn, fp, fn = tfpn_calculator(y_batch, y_pred)

                    total_test_tp += tp
                    total_test_tn += tn
                    total_test_fp += fp
                    total_test_fn += fn
            print("Testing Confusion Matrix:")
            confusion_matrix_printout(total_test_tp,
                                        total_test_tn,
                                        total_test_fp,
                                        total_test_fn)

            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            
            print(f"epoch time: {time.time()-start_time:.2f}"), 
            start_time = time.time()

            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, test Loss: {avg_test_loss:.4f}')
    
        return train_losses, test_losses
    except KeyboardInterrupt:
        return train_losses, test_losses
    

def confusion_matrix_printout(tp,tn,fp,fn):
    recall                    = np.divide(tp, (tp + fn))  if (tp + fn) != 0 else np.nan # true positive rate
    specificity               = np.divide(tn, (tn + fp))  if (tn + fp) != 0 else np.nan # true negative rate
    precision                 = np.divide(tp, (tp + fp))  if (tp + fp) != 0 else np.nan # positive predictive value
    negative_predictive_value = np.divide(tn, (tn + fn))  if (tn + fn) != 0 else np.nan # negative predictive value

    # For accuracy and f1_score, since they involve more than a simple division, we will need to handle zero division explicitly
    accuracy = (tp + tn) / (tp + tn + fp + fn)                 if (tp + tn + fp + fn)  != 0 else np.nan
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else np.nan


    t = PrettyTable(['Actual\\Pred', 'Positive', 'Negative', ''])
    t.add_row(['Positive', tp, fn, f"Recall: {recall:.2f}"])
    t.add_row(['Negative', fp, tn, f"Specificity: {specificity:.2f}"])
    t.add_row(['', f'Precision: {precision:.2f}', f'Negative predictive value: {negative_predictive_value:.2f}', f'Accuracy: {accuracy:.2f}'])

    print(t)

    ret_dict = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 
                'recall'                   : recall,
                'specificity'              : specificity,
                'precision'                : precision,
                'negative_predictive_value': negative_predictive_value,
                'accuracy'                 : accuracy,
                'f1_score'                 : f1_score}
    
    return ret_dict

def loss_n_accuracy_plt(train_losses, test_losses, train_accuracy_lst, test_accuracy_lst):
    ignore_first_n = 2
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training and evaluation loss using the left side Y axis
    # loss line should be blue and light blue
    ax1.plot(train_losses[ignore_first_n:], 'g-', label='Training Loss')
    ax1.plot(test_losses[ignore_first_n:], 'b-', label='Evaluation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Training and Evaluation Loss and Accuracy')
    ax1.legend(loc='upper left')

    # Create a second y-axis for the accuracy plots
    ax2 = ax1.twinx()
    # train
    ax2.plot(train_accuracy_lst[ignore_first_n:], 'y:o', label='Training Accuracy', linestyle='dotted')
    ax2.plot(test_accuracy_lst[ignore_first_n:], 'r:o', label='Evaluation Accuracy', linestyle='dotted')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Create a separate legend for the second y-axis
    ax2.legend(loc='upper right')

    plt.show()

train_losses, test_losses, train_cm_dicts, test_cm_dicts = train_model(model)
train_accuracy_lst = [x['accuracy'] for x in train_cm_dicts]
test_accuracy_lst = [x['accuracy'] for x in test_cm_dicts]
loss_n_accuracy_plt(train_losses,test_losses, train_accuracy_lst, test_accuracy_lst)




# Normalize data, and save the normalization parameters for each symbol in symbols_norm_params dataframe
def normalize_raw_price(df_original):
    df = df_original.copy()
    # if normalization parameters for this symbol already exist, use them
    symbol = df.index[0][0]
    if symbol in symbols_norm_params.index:
        row = symbols_norm_params.loc[symbol]
        close_mean       = row['close_mean']
        close_std        = row['close_std']
        trade_count_mean = row['trade_count_mean']
        trade_count_std  = row['trade_count_std']
    else:
        # Calculate mean and std for 'close'
        close_mean = df['close'].mean()
        close_std  = df['close'].std()

        # Normalize 'trade_count' independently
        trade_count_mean  = df['trade_count'].mean()
        trade_count_std   = df['trade_count'].std()
        # Save normalization parameters for each symbol in symbols_norm_params dataframe
        symbols_norm_params.loc[symbol] = [close_mean, close_std, trade_count_mean, trade_count_std]
        # Save symbols_norm_params dataframe to csv file
        symbols_norm_params.to_csv(pth)
    
    for column in df.columns:
        if column not in ['close', 'trade_count']:
            df[column] = (df[column] - close_mean) / close_std
    
    df['trade_count'] = (df['trade_count'] - trade_count_mean) / trade_count_std

    return (symbol, df)


# check if file exists, if not, create dataframe
pth = "symbols_norm_params.csv"
symbols_norm_params = pd.DataFrame(columns=['symbol', 'close_mean', 'close_std', 'trade_count_mean', 'trade_count_std'], index='symbol')
if os.path.exists(pth):
    symbols_norm_params = pd.read_csv(pth, index_col='symbol')
symbols_norm_params.head()


# Step 1: Mean normalize the data
# Already done. mean_normalized_data

# Step 2: Calculate the covariance matrix
covariance_matrix = np.dot(mean_normalized_data.T, mean_normalized_data)

# Step 3: Eigen-decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Step 4: Select the k largest eigenvalues and their associated eigenvectors
k = 3  # Number of dimensions to reduce to
eigenvalue_indices = np.argsort(eigenvalues)[::-1][:k] # Sort eigenvalues from largest to smallest, and get the indices of the k largest eigenvalues
selected_eigenvalues = eigenvalues[eigenvalue_indices]
selected_eigenvectors = eigenvectors[:, eigenvalue_indices].T

# Print selected eigenvalues and eigenvectors
print("Selected Eigenvalues:")
print(selected_eigenvalues)
print("\nSelected Eigenvectors:")
print(selected_eigenvectors)