import sys

from model_generator import ModelGenerator
import torch


def main():
    x = torch.ones((1, 3, 512, 512))
    model = ModelGenerator(json_path="/home/andrewheschl/Documents/ICCV-2023-VRL/src/json_models/model_definitions"
                                     "/gated_xmodule/3.json").get_model()
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    main()
