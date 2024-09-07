from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class LogisticRegression(nn.Module):  
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

if __name__ == '__main__':
    x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.5)
    # 画图
    x1 = x[:, 0]
    x2 = x[:, 1]
    plt.scatter(x1[y==1], x2[y==1], color='blue', marker="o")
    plt.scatter(x1[y==0], x2[y==0], color='red', marker="x")
    #plt.show()

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = LogisticRegression()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(6000):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f'After {epoch} iterations, the loss is {loss.item():.3f}')