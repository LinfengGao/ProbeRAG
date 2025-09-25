import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, output_size=1):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def train_model(
        X,
        y,
        input_size,
        hidden_size1=512,
        hidden_size2=256,
        output_size=1,
        num_epochs=10,
        batch_size=32,
        lr=0.001,
        report_to="tensorboard",
        log_dir="logs/classifier_training_logs/0518/llama3.1-8b",
        device="cpu"
    ):

    model = Classifier(input_size, hidden_size1, hidden_size2, output_size).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if report_to == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_graph(model, X[0].unsqueeze(0).to(device))
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for step, (batch_X, batch_y) in enumerate(dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            probs = model(batch_X)
            batch_y = batch_y.view(-1, 1)
            loss = criterion(probs, batch_y)
            accuracy = ((probs > 0.5).float() == batch_y).float().mean()
            if report_to == "tensorboard":
                writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + step)
                writer.add_scalar("Accuracy/train", accuracy.item(), epoch * len(dataloader) + step)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
        
        epoch_loss /= len(dataloader)
        epoch_accuracy /= len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
    return model


if __name__ == "__main__":
    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000, 1)).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trained_model = train_model(X, y, input_size=10, num_epochs=10, batch_size=32, lr=0.001, device=device)

    trained_model.eval()
    with torch.no_grad():
        test_X = torch.randn(10, 10).to(device)
        predictions = trained_model(test_X)
        print("Predictions:", predictions)