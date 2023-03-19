import torch
from torch import nn

       
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1) -> None:
        super().__init__()
        self.log_reg = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )  # Linear architecture, signmoid activation
            
    def forward(self, x):
        return  self.log_reg(x)  


model = LogisticRegression(16) # 16 input vars
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# example of learning 
for t in range (n_epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

