import os
import torch

class Train():

    def __init__(self, device='cpu', class_idx=None):
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion.to(self.device)
        self.class_to_idx = class_idx
    
    def train(self, model, train_loader, val_loader, optimizer, epochs=4):
    
        model.to(self.device)
        
        for i in range(epochs):
            print("="*20)
            print(f"Epoch: {i+1}/{epochs}")
            
            model.train()
            
            training_loss = 0
            for train_step, (images, labels) in enumerate(train_loader):
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                preds = model(images)
                loss = self.criterion(preds, labels)
                loss.backward()
                optimizer.step()
                
                training_loss += loss.item()

                if train_step % len(train_loader) == 0:
                    
                    model.eval()
                    with torch.no_grad():
                        validation_loss, val_accuracy = self.validation(model, val_loader)
                    
                    print(f"Step: {train_step}/{len(train_loader)} \n\t"
                        f"Validation Loss: {validation_loss:.3f} |",
                        f"Validation Accuracy: {val_accuracy:.3f}")
                    
                    training_loss = 0
                    model.train()

    def validation(self, model, val_loader):
    
        val_loss = 0
        accuracy = 0
        
        for val_step, (images, labels) in enumerate(val_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            preds = model(images)
            val_loss += self.criterion(preds, labels).item()
            
            accuracy += (preds.argmax(1) == labels).type(torch.float).sum().item()
            
        return val_loss, accuracy/len(val_loader.dataset)
    

    def test(self, model, test_loader):
        model.eval()
        model.to(self.device)
        
        with torch.no_grad():
            accuracy = 0
            
            for images, labels in iter(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                pred = model(images)
                accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()
                
        print(f"Test Accuracy: {accuracy/len(test_loader.dataset):.3f}") 


    def save_checkpoint(self, model, name):
        if not os.path.exists('models'):
            os.mkdir('models')
        
        model.class_to_idx = self.class_to_idx
        model = model.to('cpu')
        
        torch.save(model, f'models/{name}.pth')
        print(f'Saved to models/{name}.pth')

    
    def load_model(filepath):
        model = torch.load(filepath)
        return model