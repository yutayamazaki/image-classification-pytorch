import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from data_utils import load_kmnist
from trainers import RecognitionTrainer

BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4


class KMNISTDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _gray_to_rgb(gray):
        """ Convert image shape from (H, W) to (3, H, W). """
        # gray = Image.fromarray(np.uint8(gray))
        # gray = gray.resize((224, 224))
        # gray = np.asarray(gray)
        gray = torch.from_numpy(gray)
        gray_3d = gray.unsqueeze(0)
        rgb = gray_3d.repeat(3, 1, 1)
        return rgb

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img_rgb = self._gray_to_rgb(img).float()
        label = torch.tensor(label).item()
        return img_rgb, label


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_kmnist('../datasets/')
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train)

    dtrain = KMNISTDataset(X_train, y_train)
    dvalid = KMNISTDataset(X_valid, y_valid)
    dtest = KMNISTDataset(X_test, y_test)

    train_loader = DataLoader(dtrain, batch_size=BATCH_SIZE, drop_last=True)
    valid_loader = DataLoader(dvalid, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dtest, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.resnet18(True)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=LR,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)

    criterion = torch.nn.CrossEntropyLoss()

    trainer = RecognitionTrainer(model, optimizer, criterion, NUM_CLASSES)

    for epoch in range(1, 1 + NUM_EPOCHS):
        train_loss, train_acc = trainer.epoch_train(train_loader)
        valid_loss, valid_acc = trainer.epoch_eval(valid_loader)

        print(f'EPOCH: [{epoch}/{NUM_EPOCHS}]')
        print(f'TRAIN LOSS: {train_loss:.4f}, TRAIN ACC: {train_acc:.3f}')
        print(f'VALID LOSS: {valid_loss:.4f}, VALID ACC: {valid_acc:.3f}')

        params = trainer.weights
        torch.save(params, f'../{epoch}.pth')
