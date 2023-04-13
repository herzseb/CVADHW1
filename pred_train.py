import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import random
from matplotlib import pyplot as plt

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(model, dataloader, criterion_MAE, criterion_CE):
    """Validate model performance on the validation dataset"""
    running_loss = 0
    lane_dist_losses = 0
    lane_angle_losses = 0
    tl_dist_losses = 0
    tl_state_losses = 0
    model.eval()
    memory = torch.zeros(64, 9, 1000)
    memory = memory.to(device)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img, labels = data
            lane_dist = torch.unsqueeze(
                labels["lane_dist"], 1).to(dtype=torch.float32)
            lane_angle = torch.unsqueeze(
                labels["lane_angle"], 1).to(dtype=torch.float32)
            tl_dist = torch.unsqueeze(
                labels["tl_dist"], 1).to(dtype=torch.float32)
            tl_state = torch.unsqueeze(
                labels["tl_state"], 1).to(dtype=torch.float32)
            regression_target = torch.concat(
                (lane_dist, lane_angle, tl_dist), dim=1)

            img = img.to(device)
            outputs, hidden_memory = model(
                img=img, command=labels["command"], memory=memory)
            memory = hidden_memory[:, :9, :]
            regs = outputs[0].to('cpu')
            clas = outputs[1].to('cpu')
            loss = criterion_MAE(regs, regression_target)
            loss += criterion_CE(clas,
                                 torch.flatten(tl_state).to(dtype=torch.long))
            running_loss += loss.item()
            lane_dist_losses += criterion_MAE(regs[:, 0], regression_target[0])
            lane_angle_losses += criterion_MAE(
                regs[:, 1], regression_target[1])
            tl_dist_losses += criterion_MAE(regs[:, 2], regression_target[2])
            tl_state_losses += criterion_CE(clas, tl_state)
        running_loss = running_loss/(i * img.size()[0])
        lane_dist_losses = lane_dist_losses/(i * img.size()[0])
        lane_angle_losses = lane_angle_losses/(i * img.size()[0])
        tl_dist_losses = tl_dist_losses/(i * img.size()[0])
        tl_state_losses = tl_state_losses/(i * img.size()[0])
        return running_loss, lane_dist_losses, lane_angle_losses, tl_dist_losses, tl_state_losses


def train(model, iters, optimizer, criterion_MAE, criterion_CE):
    """Train model on the training dataset for one epoch"""
    running_loss = 0.0
    iter = 0
    left_fin, right_fin, straight_fin, followlane_fin = False, False, False, False
    memory = torch.zeros(64, 9, 1000)
    memory = memory.to(device)
    while True:
        try:
            curr_iter = random.choices(iters, weights=(1, 1, 1, 30))[0]
            img, labels = next(curr_iter)
            lane_dist = torch.unsqueeze(
                labels["lane_dist"], 1).to(dtype=torch.float32)
            lane_angle = torch.unsqueeze(
                labels["lane_angle"], 1).to(dtype=torch.float32)
            tl_dist = torch.unsqueeze(
                labels["tl_dist"], 1).to(dtype=torch.float32)
            tl_state = torch.unsqueeze(
                labels["tl_state"], 1).to(dtype=torch.float32)
            regression_target = torch.concat(
                (lane_dist, lane_angle, tl_dist), dim=1)
            optimizer.zero_grad()

            img = img.to(device)
            outputs, hidden_memory = model(
                img=img, command=labels["command"], memory=memory)
            memory = hidden_memory[:, :9, :]
            regs = outputs[0].to('cpu')
            clas = outputs[1].to('cpu')
            loss = criterion_MAE(regs, regression_target)
            loss += criterion_CE(clas,
                                 torch.flatten(tl_state).to(dtype=torch.long))
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()
            iter = iter + 1

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

    avg_loss = running_loss/(iter * img.size()[0])
    print(avg_loss)
    return avg_loss


def plot_losses(train_loss, val_loss, lane_dist_losses, lane_angle_losses, tl_dist_losses, tl_state_losses):
    """Visualize your plots and save them for your report."""
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.plot(lane_dist_losses, label='lane_dist_losses')
    plt.plot(lane_angle_losses, label='lane_angle_losses')
    plt.plot(tl_dist_losses, label='tl_dist_losses')
    plt.plot(tl_state_losses, label='tl_state_losses')
    plt.legend()
    plt.savefig("affordance_plt")
    pass


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/userfiles/ssafadoust20/expert_data/train"
    val_root = "/userfiles/ssafadoust20/expert_data/val"
    # train_root = "C:\\Users\\User\\Desktop\\expert_data\\expert_data\\train\\"
    # val_root = "C:\\Users\\User\\Desktop\\expert_data\\expert_data\\val\\"
    model = AffordancePredictor().to(device)
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
    save_path = "affordance_model.ckpt"
    checkpoint = "affordance_checkpoint.pt"

    train_loader_left = DataLoader(train_dataset_left, batch_size=batch_size, shuffle=True,
                                   drop_last=True)
    train_loader_right = DataLoader(train_dataset_right, batch_size=batch_size, shuffle=True,
                                    drop_last=True)
    train_loader_straight = DataLoader(train_dataset_staright, batch_size=batch_size, shuffle=True,
                                       drop_last=True)
    train_loader_followlane = DataLoader(train_dataset_lanefollow, batch_size=batch_size, shuffle=True,
                                         drop_last=True)

    loader_iter_left = iter(train_loader_left)
    loader_iter_right = iter(train_loader_right)
    loader_iter_straight = iter(train_loader_straight)
    loader_iter_followlane = iter(train_loader_followlane)

    iters = [loader_iter_left, loader_iter_right,
             loader_iter_straight, loader_iter_followlane]
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion_MAE = torch.nn.L1Loss()
    criterion_CE = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    train_losses = []
    val_losses = []
    lane_dist_losses = []
    lane_angle_losses = []
    tl_dist_losses = []
    tl_state_losses = []
    best_val_loss = 10000
    early_stopper = 0
    for i in range(num_epochs):
        train_losses.append(train(model, iters, optimizer,
                            criterion_MAE, criterion_CE))
        running_loss, lane_dist_losses, lane_angle_losses, tl_dist_losses, tl_state_losses = validate(model, val_loader,
                                                                                                      criterion_MAE, criterion_CE)
        val_losses.append(running_loss)
        lane_dist_losses.append(lane_dist_losses)
        lane_angle_losses.append(lane_angle_losses)
        tl_dist_losses.append(tl_dist_losses)
        tl_state_losses.append(tl_state_losses)
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
    plot_losses(train_losses, val_losses, lane_dist_losses,
                lane_angle_losses, tl_dist_losses, tl_state_losses)


if __name__ == "__main__":
    main()
