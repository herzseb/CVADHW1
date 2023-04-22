import os
import torch
from torchvision import transforms

import yaml

from carla_env.env import Env

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Evaluator():
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent = self.load_agent()
        self.totensor = transforms.ToTensor()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def load_agent(self):
        self.model = torch.load("cilrs_model.ckpt")
        self.model.eval()

    def generate_action(self, rgb, command, speed):
        # preprocess images to fit pretrained resnet 
        # batch image and speed 
        rgb = self.totensor(rgb)
        rgb = rgb[:,90:400,:]
        rgb = self.preprocess(rgb)
        rgb = rgb.to(device)
        rgb = torch.unsqueeze(rgb, dim=0)
        speed = torch.tensor(speed, dtype=torch.float32)
        speed = speed.to(device)
        speed = torch.unsqueeze(speed, dim=0)
        speed = torch.unsqueeze(speed, dim=0)
        out = self.model(rgb, [command], speed)
        out = out.to('cpu')
        out = torch.squeeze(out)
        return out[0].item(), out[1].item(), out[2].item()

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, brake, steer = self.generate_action(rgb, command, speed)
        brake = 0 if brake < 0.01 else brake
        print("------------------")
        print("command", command)
        print("speed", speed)
        action = {
            "throttle": throttle,
            "brake": brake,
            "steer": steer
        }
        print(action)
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            state, _, is_terminal = self.env.reset()
            for i in range(5000):
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
