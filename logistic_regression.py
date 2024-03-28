from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import torch


class LogisticRegression(torch.nn.Module):  
  def __init__(self):
    super(LogisticRegression, self).__init__()
    self.linear = torch.nn.Linear(2, 1, bias=True)  

  def forward(self, x):
    return torch.sigmoid(self.linear(x))

def compute_loss(y_pred, y):
  y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
  loss = -torch.mean(y * torch.log(y_pred) + (1-y) * torch.log(1-y_pred))
  return loss
  
if __name__ == '__main__':
  x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.5)
  # 画图
  x1 = x[:, 0]
  x2 = x[:, 1]
  plt.scatter(x1[y==1], x2[y==1], color='blue', marker="o")
  plt.scatter(x1[y==0], x2[y==0], color='red', marker="x")
  x1 = torch.tensor(x1, dtype=torch.float32)
  x2 = torch.tensor(x2, dtype=torch.float32)
  
  x = torch.tensor(x, dtype=torch.float32)
  y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
  
  model = LogisticRegression()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  
  for epoch in range(6000):
    y_pred = model(x)
    loss = compute_loss(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
      print(f'After {epoch} iterations, the loss is {loss.item():.3f}')
