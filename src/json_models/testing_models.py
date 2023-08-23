import sys
sys.path.append("/home/andrew.heschl/Documents/ICCV-2023-VRL")
from src.json_models.src.model_generator import ModelGenerator
import torch


def main():
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(device)
    x = torch.ones((1, 3, 480, 480)).to(device)
    model = ModelGenerator(json_path="/home/andrew.heschl/Documents/ICCV-2023-VRL/src/json_models/model_definitions"
                                     "/env2/k37.json").get_model().to(device)
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    main()
