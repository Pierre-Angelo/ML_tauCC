import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

a = 0.5
b = 0.2

start = 0
end = 1
step = 0.01

X = torch.arange(start,end,step).unsqueeze(dim = 1)

Y = a * X + b

train_split = int(0.8 * len(X))

input_train, train_labels = X[:train_split], Y[:train_split]
input_test,test_labels = X[train_split:], Y[train_split:]

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14})


class LinRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(in_features=1, out_features=1)
        self.loss_func = nn.L1Loss()
        self.optim = torch.optim.Adam(self.parameters(),lr = 0.1 )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim,0.95)

    def forward(self, input):
       return self.layer(input)
    
    def fit(self, input, labels, input_test, test_labels, epochs):
        loss_values = {"train" : [],"test" : []}
        for epoch in range(epochs):
            self.train()

            output = self(input)

            loss = self.loss_func(output,labels)
            loss_values["train"].append(loss.item())

            self.optim.zero_grad()

            loss.backward()

            self.optim.step()

            self.scheduler.step()
            

            #test
            self.eval()

            with torch.inference_mode():
                output_test = self(input_test)

                loss_test = self.loss_func(output_test,test_labels)
               
            loss_values["test"].append(loss_test.item())

        return loss_values
            

    
torch.manual_seed(0)

epochs = 300

model0 = LinRegModel()
model0.to(device)

input_train, train_labels = input_train.to(device), train_labels.to(device)
input_test,test_labels = input_test.to(device),test_labels.to(device)

loss_stats = model0.fit(input_train,train_labels, input_test, test_labels, epochs)

plt.plot(loss_stats["train"], label="Train loss")
plt.plot(loss_stats["test"], label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

model0.eval()
with torch.inference_mode():
   pred = model0(input_test)

plot_predictions(input_train,train_labels,input_test,test_labels, pred)

print(model0.state_dict())

"""
# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 

lmodel0 = LinRegModel()
lmodel0.load_state_dict(torch.load(MODEL_SAVE_PATH))

lmodel0.eval()

with torch.inference_mode():
   lpred = lmodel0(input_test)

plot_predictions(input_train,train_labels,input_test,test_labels, lpred)
"""
plt.show()