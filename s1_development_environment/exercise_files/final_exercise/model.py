from torch import nn
import torch

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # declare layers
                
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(784, 128)
        
        # Hidden laayer with 128 units
        self.fc2 = nn.Linear(128,64)
        #Hidden layer with 64 units
        self.fc3 = nn.Linear(64, 64)
        
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)
        
        # Define RELU activation and softmax output 
        self.softmax = nn.Softmax(dim=1)
        self.relu = F.relu
        
    def forward(self, x):
		x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.output(x)
        x= self.softmax(x)
		# declare how to pass through
		return self.fc1(x)
		
		
if __name__ == "__main__":
	model = MyAwesomeModel()
	out = model(torch.randn(2, 10))
	assert out.shape[1] == 5 
