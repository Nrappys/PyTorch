import numpy as np
import matplotlib.pyplot as plt
import torch

z = np.arange(2).repeat(40)
r = np.random.normal(z+1, 0.25)
t = np.random.uniform(0, np.pi, 80)
x = r*np.cos(t)
y = r*np.sin(t)
X = np.array([x,y]).T

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.mps.is_available()
                      else "cpu")
print(f"Using device: {device}")
linear_layer = torch.nn.Linear(2, 3).to(device)   

class testnueralnet(torch.nn.Module):
    def __init__(self):
        super(testnueralnet, self).__init__()
        self.linearlayer1 = torch.nn.Linear(2, 3)
        self.linearlayer2 = torch.nn.Linear(3, 3)
        self.linearlayer3 = torch.nn.Linear(3, 2)

    def forward(self, x):
        h1 = self.linearlayer1(x)
        a1 = torch.nn.functional.relu(h1)
        h2 = self.linearlayer2(a1)
        a2 = torch.nn.functional.relu(h2)
        h3 = self.linearlayer3(a2)
        return h3

X = torch.tensor(X, dtype=torch.float32).to(device)
z = torch.tensor(z).to(device)
nn = testnueralnet().to(device)

optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)
cross_entropy = torch.nn.CrossEntropyLoss()
for epoch in range(200):
    optimizer.zero_grad()
    output = nn(X)
    loss = cross_entropy(output, z)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")