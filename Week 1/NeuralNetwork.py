import torch
import numpy as np
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
nn = testnueralnet().to(device)
input = torch.tensor(np.array([[0.5, 0.2], [0.1, 0.4]]), dtype=torch.float32).to(device)
output = nn(input)
print(output)