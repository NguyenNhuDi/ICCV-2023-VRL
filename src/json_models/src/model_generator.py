import sys
sys.path.append("/home/andrew.heschl/Documents/ICCV-2023-VRL/src")
import json
from src.json_models.src.model_builder import ModelBuilder
import torch.nn as nn


class ModelGenerator:
    def __init__(self, json_path: str) -> None:
        """
        Given a path to a json model skeleton, helps builds a model, and verifies that the json is correct.
        """
        print(json_path)
        self.json_path = json_path
        self._build_architecture()

    def _build_architecture(self) -> None:
        """
        Verifies the json structure, and then creates model.
        """
        with open(self.json_path) as file:
            model_definition = json.load(file)

        if "LogKwargs" in model_definition.keys():
            self.log_kwargs = model_definition.pop('LogKwargs')
        else:
            self.log_kwargs = None

        if "Encoder" in model_definition.keys():
            ModelGenerator.verify_unet_structure(model_definition)
            model_definition = ModelGenerator._convert_unetlike_to_normal(model_definition)
        else:
            ModelGenerator.verify_structure(model_definition)
        model = ModelBuilder(model_definition['Tag'], model_definition['Children'])

        self.model = model

    def get_model(self) -> nn.Module:
        """
        Returns the generated model.
        """
        return self.model

    def get_log_kwargs(self) -> dict:
        return self.log_kwargs

    @staticmethod
    def _convert_unetlike_to_normal(model_definition: dict) -> dict:

        encoder_elements = model_definition['Encoder']
        middle_elements = model_definition['Middle']
        decoder_elements = model_definition['Decoder']

        sequential = []
        sequential += encoder_elements
        sequential += middle_elements
        sequential += decoder_elements

        new_root = {
            "Tag": "Parent",
            "Children": sequential
        }

        return new_root

    @staticmethod
    def verify_structure(model_definition: dict) -> bool:
        """
        Verifies that the structure of a json model is valid.
        Returns true if so, raises an exception otherwise.
        """
        keys = model_definition.keys()

        ModelGenerator._validatenode(list(keys))

        if 'Tag' in keys:
            if not isinstance(model_definition['Tag'], str):
                raise InvalidJsonArchitectureException("'Tag' must be an instance of 'str'")

        if 'Children' in keys:
            if not isinstance(model_definition['Children'], list):
                raise InvalidJsonArchitectureException("'Children' must be an instance of 'list'")

        if 'ComponentClass' in keys:
            if not isinstance(model_definition['ComponentClass'], str):
                raise InvalidJsonArchitectureException("'ComponentClass' must be an instance of 'str'")

        if 'args' in keys:
            if not isinstance(model_definition['args'], dict):
                raise InvalidJsonArchitectureException("'args' must be an instance of 'dict'")

        if 'store_out' in keys:
            if not isinstance(model_definition['store_out'], str):
                raise InvalidJsonArchitectureException("'store_out' must be an instance of 'str'")

        if 'Tag' not in keys and 'ComponentClass' not in keys:
            raise InvalidJsonArchitectureException("You didn't define 'ComponentClass'")

        if 'Children' in keys:
            for child in list(model_definition['Children']):
                ModelGenerator.verify_structure(child)

        return True

    @staticmethod
    def verify_unet_structure(model_definition: dict) -> bool:
        """
        Verifies that the structure of a json model is valid.
        Returns true if so, raises an exception otherwise.
        """
        keys = model_definition.keys()
        if not ('Encoder' in keys and 'Decoder' in keys and 'Middle' in keys) or len(keys) != 3:
            raise InvalidJsonArchitectureException("When defining a UNet like structure (you defined 'Encoder')" +
                                                   " you must define exactly: Encoder, Decoder, and Middle")
        if not (
                isinstance(model_definition['Encoder'], list) and
                isinstance(model_definition['Encoder'], list) and
                isinstance(model_definition['Encoder'], list)):
            raise InvalidJsonArchitectureException("When defining a UNet like structure (you defined 'Encoder')" +
                                                   " the portions should be defined as lists!")

        for element in model_definition['Encoder']:
            ModelGenerator.verify_structure(element)
        for element in model_definition['Decoder']:
            ModelGenerator.verify_structure(element)
        for element in model_definition['Middle']:
            ModelGenerator.verify_structure(element)

        return True

    @staticmethod
    def _validatenode(keys: list) -> bool:
        """
        Verifies that the keys of a node are valid.
        Returns true if so, raises an exception otherwise.
        """
        if 'Tag' in keys and 'Children' not in keys:
            raise InvalidJsonArchitectureException("You must define 'Tag' and 'Children' together.")
        if 'Tag' in keys and 'args' in keys:
            raise InvalidJsonArchitectureException("You cannot define 'Tag' and 'args' at the same level.")
        if 'ComponentClass' in keys and 'args' not in keys:
            raise InvalidJsonArchitectureException("You must define 'ComponentClass' and 'args' together.")
        return True


class InvalidJsonArchitectureException(Exception):
    """
    Exception raised when a user tries to build a model with invalid json.
    """

    def __init__(self, message="Invalid model architecture in json."):
        self.message = message
        super().__init__(self.message)


if __name__ == "__main__":
    factory = ModelGenerator("/home/andrewheschl/Documents/3DSegmentation/src/models/model_definitions/unetlike.json")
