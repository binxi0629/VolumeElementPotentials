import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

'''
Subnet architecture

c (int): base number of units, scaled by different integers in hidden layers
'''
class SubNet(nn.Module):
    def __init__(self, input_size, c=3, stress_fit=False):
        super().__init__()
        self.FC1 = nn.Linear(input_size, c).double()
        self.FC2 = nn.Linear(c, 2*c).double()
        self.FC3 = nn.Linear(2 * c, c).double()

        self.FC4 = nn.Linear(c, 3).double() if stress_fit else nn.Linear(c, 1).double()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):

        x = self.FC1(x)
        x = self.leaky_relu(x)

        x = self.FC2(x)
        x = self.leaky_relu(x)

        x = self.FC3(x)
        x = self.leaky_relu(x)

        x = self.FC4(x)
        return x


class ModelContainer():
    def __init__(self, model, stress_fit=False):
        self.model = model
        self.stress_fit = stress_fit
        self.criterion = torch.nn.MSELoss()
    
    def fit(self, loader, ep, learning_rate=1e-3):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.model.train()

        if self.stress_fit:
            for epoch in range(ep):
                f_loss_sum, e_loss_sum, loss_sum = 0, 0, 0

                for X, forces, energy in loader:

                    energy = energy.float()
                    self.optimizer.zero_grad()
                    if np.nan in energy: continue
                    tot_e = torch.zeros(energy.size())

                    for i in range(len(X[0])):

                        res = self.model(X[:, i, :])
                        tot_e += res[:, 2]

                    out_forces = self.model(X)[:, :, 0:2]

                    f_loss = torch.nn.functional.mse_loss(out_forces, forces)
                    e_loss = self.criterion(tot_e.squeeze(), energy)
                    loss = f_loss + e_loss

                    loss.backward()

                    f_loss_sum += f_loss.item()
                    e_loss_sum += e_loss.item()
                    loss_sum += loss.item()
                    self.optimizer.step()
                self.scheduler.step()
                print("Epoch: ", epoch, "Energy Loss (MSE): ", np.round(e_loss_sum / len(loader), 6), "Forces Loss (MSE): ", np.round(f_loss_sum/len(loader),6))
        else:
            for epoch in range(ep):
                loss_sum = 0
                for X, y in loader:
                    y = y.float()
                    self.optimizer.zero_grad()
                    if np.nan in y: continue
                    tot_e = torch.zeros(y.size()).unsqueeze(1)
                    for i in range(len(X[0])):
                        tot_e += self.model(X[:,i,:])
                        # print("y_pred, y_0", tot_e)

                    loss = self.criterion(tot_e.squeeze(), y)
                    loss.backward()

                    loss_sum += loss.item()
                    # nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()
                self.scheduler.step()
                print("Epoch: ", epoch, "Loss (MSE): ", loss_sum / len(loader))
        return


    def predict(self, loader, save_sub_preds=False):
        truth, pred, sub_pred = [], [], []
        self.model.eval()
        loss_sum = 0
        for X, y in loader:
            y = y.float()
            if np.nan in y: continue
            tot_e = torch.zeros(y.size()).unsqueeze(1)
            with torch.no_grad():
                if save_sub_preds: batch_sub_pred = []
                for i in range(len(X[0])):
                    sub_e = self.model(X[:,i,:])
                    tot_e += sub_e
                    if save_sub_preds: batch_sub_pred.append(sub_e.squeeze().tolist())
                if save_sub_preds: sub_pred.extend(np.array(batch_sub_pred).T.flatten())
                loss = self.criterion(tot_e.squeeze(), y)
                truth.extend(y.tolist())
                pred.extend(tot_e.squeeze().tolist())
                # print("Truth: ", list(y))
                # print("Prediction: ", list(tot_e.squeeze()))
            loss_sum += loss.item()
        print("Test MSELoss", loss_sum / len(loader))
        if save_sub_preds:
            return truth, pred, sub_pred
        return truth, pred
