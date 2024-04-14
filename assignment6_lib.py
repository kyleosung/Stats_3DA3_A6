import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(24, 40)
        self.fc2 = nn.Linear(40, 18)
        self.fc3 = nn.Linear(18, 1)
        self.dropout = nn.Dropout(p=0.05)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.000025)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.000025)
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))



def train(model, criterion, optimizer, EPOCHS, X_filled, y_r):
    training_loss_hist = []
    validation_loss_hist = []
    accuracy_history = []
    print('Begin Training!')
    for epoch in range(EPOCHS):
        try:

            # Use Cross-Validation with different train-test splits per epoch
            X_train_DL, X_test_DL, y_train_DL, y_test_DL = train_test_split(
                X_filled,
                y_r,
                train_size = 0.95,
                test_size = 0.05, # train is 70%, test is 30% 
                random_state = epoch, # random seed = 1
                stratify = y_r,
            )
            X_train_torch = torch.tensor(X_train_DL, dtype=torch.float32)
            X_test_torch = torch.tensor(X_test_DL, dtype=torch.float32)
            y_train_torch = torch.tensor(y_train_DL.to_numpy().ravel(), dtype=torch.float32).flatten()
            y_test_torch = torch.tensor(y_test_DL.to_numpy().ravel(), dtype=torch.float32).flatten()

            ## TRAINING MODE
            model.train()

            optimizer.zero_grad()
            outputs = model(X_train_torch).flatten()
            loss = criterion(outputs, y_train_torch)
            loss.backward()

            # model suffers from exploding gradient problem. we clip the gradient at 0.9
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.95)
            optimizer.step()

            ## VALIDATION MODE
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_torch)
                val_loss = criterion(val_outputs.flatten(), y_test_torch.flatten())

            all_outputs = torch.hstack((val_outputs.flatten(), outputs))
            all_y_test = torch.hstack((y_test_torch, y_train_torch))

            accuracy = accuracy_score(all_outputs >= 0.5, all_y_test)

            if (epoch+1) % 20 == 0:
                print(f'Epoch {epoch+1}. Training Loss: {loss:.3f}. Validation Loss: {val_loss:.3f}. Accuracy: {accuracy:.3f}.')
            
            training_loss_hist.append(float(loss))
            validation_loss_hist.append(float(val_loss))
            accuracy_history.append(accuracy)

        except KeyboardInterrupt:
            print("Manual stop: Ended training early!")
            break

        if accuracy >= 0.95 and epoch > 2000:
            print(f'Epoch {epoch+1}. Training Loss: {loss:.3f}. Validation Loss: {val_loss:.3f}. Accuracy: {accuracy:.3f}.')
            print('Training complete! Early convergence.')
            break
        
    return training_loss_hist, validation_loss_hist, accuracy_history



def get_current_version():
    from pathlib import Path
    import pickle

    model_file_path = 'models/model1.pth'
    counter = 1

    while Path(model_file_path).is_file(): # ensure that no files are overwritten
        counter += 1
        model_file_path = f'models/model{counter}.pth'
    
    return counter