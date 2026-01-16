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
z = torch.tensor(z).to(device)
model = NormalNN().to(device)

best = float('inf')
patience = 7
bad = 0

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
cross_entropy = torch.nn.CrossEntropyLoss()

for epoch in range(200):
    optimizer.zero_grad()
    output = model(X)
    loss = cross_entropy(output, z)
    loss.backward()
    optimizer.step()
    val_loss = loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")
    
    
    if val_loss < best:
        best = val_loss
        bad = 0
        print(f"Saving model at epoch {epoch+1} with loss {best:.4f}")
        torch.save(model.state_dict(), 'Week 1\\best_model.pth')
    else:
        bad += 1
        if bad >= patience:
            break
    
model.eval()  
with torch.no_grad():
    out = model(X)
    preds = torch.argmax(out, dim=1)
    acc = (preds == z).float().mean().item()
    print(f'Accuracy: {acc*100:.2f}%')