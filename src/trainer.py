import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class AbstractTrainer:

    def __init__(self, model, optimizer, criterion, num_classes):
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        self.num_classes = num_classes

    def epoch_train(self, train_loader):
        raise NotImplementedError()

    def epoch_eval(self, eval_loader):
        raise NotImplementedError()

    @property
    def weights(self):
        return self._model.state_dict()


class RecognitionTrainer(AbstractTrainer):

    def epoch_train(self, train_loader):
        self._model.train()
        epoch_loss = 0
        y_preds, y_trues = torch.Tensor([]).long(), torch.Tensor([]).long()

        for inputs, targets in tqdm(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()

            self.optimizer.zero_grad()
            outputs = self._model(inputs)

            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            y_preds = torch.cat([y_preds, predicted.cpu()])
            y_trues = torch.cat([y_trues, targets.cpu()])

        epoch_loss /= len(train_loader)
        y_trues = y_trues.numpy()
        y_preds = y_preds.numpy()
        acc = accuracy_score(y_trues, y_preds)

        return epoch_loss, acc

    def epoch_eval(self, eval_loader):
        self._model.eval()
        epoch_loss = 0
        y_preds, y_trues = torch.Tensor([]).long(), torch.Tensor([]).long()

        for inputs, targets in tqdm(eval_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()
            outputs = self._model(inputs)

            loss = self.criterion(outputs, targets)

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            y_preds = torch.cat([y_preds, predicted.cpu()])
            y_trues = torch.cat([y_trues, targets.cpu()])

        epoch_loss /= len(eval_loader)
        y_trues = y_trues.numpy()
        y_preds = y_preds.numpy()
        acc = accuracy_score(y_trues, y_preds)

        return epoch_loss, acc
