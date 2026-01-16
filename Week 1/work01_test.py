import torch
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(2).repeat(40)
r = np.random.normal(z+1, 0.25)
t = np.random.uniform(0, np.pi, 80)
x = r * np.cos(t)
y = r * np.sin(t)
X = np.array([x, y]).T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class NormalNN(torch.nn.Module):
    def __init__(self):
        super(NormalNN, self).__init__()
        self.linearlayer1 = torch.nn.Linear(2, 32)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.linearlayer2 = torch.nn.Linear(32, 32)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.linearlayer3 = torch.nn.Linear(32, 2)

    def forward(self, x):
        h1 = self.linearlayer1(x)
        a1 = torch.nn.functional.relu(h1)
        a1 = self.dropout1(a1)
        h2 = self.linearlayer2(a1)
        a2 = torch.nn.functional.relu(h2)
        a2 = self.dropout2(a2)
        h3 = self.linearlayer3(a2)
        return h3

X = torch.tensor(X, dtype=torch.float32).to(device)
z = torch.tensor(z, dtype=torch.long).to(device)

model = NormalNN().to(device)
model.load_state_dict(torch.load(r'Week 1\best_model.pth', weights_only=True))
model.eval() 

print("Loaded model from best_model.pth")

with torch.no_grad():
    out = model(X)
    preds = torch.argmax(out, dim=1)
    acc = (preds == z).float().mean().item()
    print(f'Accuracy: {acc*100:.2f}%')
