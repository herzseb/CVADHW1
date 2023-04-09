import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import random
from matplotlib import pyplot as plt

from expert_dataset import ExpertDataset
from models.cilrs import CILRS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(model, dataloader, criterion, batchsize):
    """Validate model performance on the validation dataset"""
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img, labels = data
            throttle = torch.unsqueeze(
                labels["throttle"], 1).to(dtype=torch.float32)
            brake = torch.unsqueeze(labels["brake"], 1).to(dtype=torch.float32)
            steer = torch.unsqueeze(labels["steer"], 1).to(dtype=torch.float32)
            speed = torch.unsqueeze(labels["speed"], 1).to(dtype=torch.float32)

            target = torch.concat((throttle, brake, steer, speed), dim=1)
            img = img.to(device)
            speed = speed.to(device)
            outputs = model(
                img=img, command=labels["command"], measured_speed=speed)
            outputs = outputs.to('cpu')
            target = target.to('cpu')
            loss = criterion(outputs, target)
            running_loss += loss.item()
        avg_loss = running_loss/((i+1) * batchsize)
        print("avg val loss ", avg_loss)
        return avg_loss


def train(model, loaders, optimizer, criterion, batchsize):
    """Train model on the training dataset for one epoch"""
    model.train()
    running_loss = 0.0
    it = 0
    left_fin, right_fin, straight_fin, followlane_fin = False, False, False, False
    loader_iter_left = iter(loaders[0])
    loader_iter_right = iter(loaders[1])
    loader_iter_straight = iter(loaders[2])
    loader_iter_followlane = iter(loaders[3])

    iters = [loader_iter_left, loader_iter_right,
             loader_iter_straight, loader_iter_followlane]
    while True:
        try:
            curr_iter = random.choices(iters, weights=(1, 1, 1, 30))[0]
            img, labels = next(curr_iter)
            throttle = torch.unsqueeze(
                labels["throttle"], 1).to(dtype=torch.float32)
            brake = torch.unsqueeze(labels["brake"], 1).to(dtype=torch.float32)
            steer = torch.unsqueeze(labels["steer"], 1).to(dtype=torch.float32)
            speed = torch.unsqueeze(labels["speed"], 1).to(dtype=torch.float32)

            target = torch.concat((throttle, brake, steer, speed), dim=1)

            optimizer.zero_grad()

            img = img.to(device)
            speed = speed.to(device)
            outputs = model(
                img=img, command=labels["command"], measured_speed=speed)
            outputs = outputs.to('cpu')
            target = target.to('cpu')
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            it += 1
            print(it)

        except StopIteration:
            if curr_iter == iters[0]:
                left_fin = True
            elif curr_iter == iters[1]:
                right_fin = True
            elif curr_iter == iters[2]:
                straight_fin = True
            elif curr_iter == iters[3]:
                followlane_fin = True
            if left_fin and right_fin and straight_fin and followlane_fin:
                break

    avg_loss = running_loss/((it+1) * batchsize)
    print("avg train loss ", avg_loss)
    return avg_loss


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.savefig("train_plot.png")


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/userfiles/ssafadoust20/expert_data/train"
    val_root = "/userfiles/ssafadoust20/expert_data/val"
    # train_root = "C:\\Users\\User\\Desktop\\expert_data\\expert_data\\train\\"
    # val_root = "C:\\Users\\User\\Desktop\\expert_data\\expert_data\\val\\"
    model = CILRS().to(device)
    train_dataset_left = ExpertDataset(train_root, transform=True, command=0)
    train_dataset_right = ExpertDataset(train_root, transform=True, command=1)
    train_dataset_staright = ExpertDataset(
        train_root, transform=True, command=2)
    train_dataset_lanefollow = ExpertDataset(
        train_root, transform=True, command=3)
    val_dataset = ExpertDataset(val_root, transform=True)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 50
    batch_size = 64
    save_path = "cilrs_model.ckpt"
    checkpoint = "cilrs_checkpoint.pt"

    train_loader_left = DataLoader(train_dataset_left, batch_size=batch_size, shuffle=True,
                                   drop_last=True, num_workers=4)
    train_loader_right = DataLoader(train_dataset_right, batch_size=batch_size, shuffle=True,
                                    drop_last=True, num_workers=4)
    train_loader_straight = DataLoader(train_dataset_staright, batch_size=batch_size, shuffle=True,
                                       drop_last=True, num_workers=4)
    train_loader_followlane = DataLoader(train_dataset_lanefollow, batch_size=batch_size, shuffle=True,
                                         drop_last=True, num_workers=4)
    loaders = [train_loader_left, train_loader_right, train_loader_straight, train_loader_followlane]
    

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    train_losses = []
    val_losses = []
    best_val_loss = 10000
    early_stopper = 0
    for i in range(num_epochs):
        print("EPOCH ", i)
        train_losses.append(train(model, loaders, optimizer, criterion, batch_size))
        val_losses.append(validate(model, val_loader, criterion, 1))
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint)
        if val_losses[-1] < best_val_loss:
            torch.save(model, save_path)
            early_stopper = 0
            best_val_loss = val_losses[-1]
        else:
            early_stopper += 1
        if early_stopper >= 10:
            break
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
