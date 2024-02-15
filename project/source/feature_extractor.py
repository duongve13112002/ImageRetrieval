from transformers import ViTModel,AutoImageProcessor,BeitModel,MobileViTV2Model,BitModel,MobileNetV2Model,ResNetModel,EfficientNetModel,EfficientFormerModel
import torch
import numpy as np

def number_dimention_to_global_average(tensor_arr):
    number_remove = len(tensor_arr.shape) - 2
    if number_remove == 0:
        return tensor_arr
    elif number_remove == 1:
        return tensor_arr.mean(dim=1)
    else:
        lst = [2]
        number_remove -=1
        while number_remove > 0:
            lst.append(lst[-1]+1)
            number_remove -=1
        dimention = tuple(lst)
        return tensor_arr.mean(dim=dimention)

class VisionTransformer:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "google/vit-base-patch16-224-in21k"
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)

    def extract(self, image) -> np.ndarray:

        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feature = self.model(**inputs)
        feature = feature.last_hidden_state# Extract the hidden states from the last encoder layer

        # Extract specific features based on your needs
        feature = number_dimention_to_global_average(feature) # Average pooling across all tokens

        if torch.cuda.is_available():
            feature = feature.cpu().numpy()
        else:
            feature = feature.numpy()

        feature = feature.reshape((-1,))
        return feature

class BEiT:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "microsoft/beit-large-patch16-224-pt22k-ft22k"
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model = BeitModel.from_pretrained(model_name).to(self.device)

    def extract(self, image) -> np.ndarray:

        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feature = self.model(**inputs)
        feature = feature.last_hidden_state# Extract the hidden states from the last encoder layer

        # Extract specific features based on your needs
        feature = number_dimention_to_global_average(feature) # Average pooling across all tokens

        if torch.cuda.is_available():
            feature = feature.cpu().numpy()
        else:
            feature = feature.numpy()

        feature = feature.reshape((-1,))
        return feature

class MobileViTV2:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "apple/mobilevitv2-1.0-imagenet1k-256"
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model = MobileViTV2Model.from_pretrained(model_name).to(self.device)

    def extract(self, image) -> np.ndarray:

        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feature = self.model(**inputs)
        feature = feature.last_hidden_state# Extract the hidden states from the last encoder layer

        # Extract specific features based on your needs
        feature = number_dimention_to_global_average(feature) # Average pooling across all tokens

        if torch.cuda.is_available():
            feature = feature.cpu().numpy()
        else:
            feature = feature.numpy()

        feature = feature.reshape((-1,))
        return feature

class Bit:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "google/bit-50"
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model = BitModel.from_pretrained(model_name).to(self.device)

    def extract(self, image) -> np.ndarray:

        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feature = self.model(**inputs)
        feature = feature.last_hidden_state# Extract the hidden states from the last encoder layer

        # Extract specific features based on your needs
        feature = number_dimention_to_global_average(feature) # Average pooling across all tokens

        if torch.cuda.is_available():
            feature = feature.cpu().numpy()
        else:
            feature = feature.numpy()

        feature = feature.reshape((-1,))
        return feature

class EfficientFormer:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "snap-research/efficientformer-l1-300"
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model = EfficientFormerModel.from_pretrained(model_name).to(self.device)

    def extract(self, image) -> np.ndarray:

        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feature = self.model(**inputs)
        feature = feature.last_hidden_state# Extract the hidden states from the last encoder layer

        # Extract specific features based on your needs
        feature = number_dimention_to_global_average(feature) # Average pooling across all tokens

        if torch.cuda.is_available():
            feature = feature.cpu().numpy()
        else:
            feature = feature.numpy()

        feature = feature.reshape((-1,))
        return feature


class MobileNetV2:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "google/mobilenet_v2_1.0_224"
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model = MobileNetV2Model.from_pretrained(model_name).to(self.device)

    def extract(self, image) -> np.ndarray:

        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feature = self.model(**inputs)
        feature = feature.last_hidden_state# Extract the hidden states from the last encoder layer

        # Extract specific features based on your needs
        feature = number_dimention_to_global_average(feature) # Average pooling across all tokens

        if torch.cuda.is_available():
            feature = feature.cpu().numpy()
        else:
            feature = feature.numpy()

        feature = feature.reshape((-1,))
        return feature


class ResNet:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "microsoft/resnet-50"
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetModel.from_pretrained(model_name).to(self.device)

    def extract(self, image) -> np.ndarray:

        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feature = self.model(**inputs)
        feature = feature.last_hidden_state# Extract the hidden states from the last encoder layer

        # Extract specific features based on your needs
        feature = number_dimention_to_global_average(feature) # Average pooling across all tokens

        if torch.cuda.is_available():
            feature = feature.cpu().numpy()
        else:
            feature = feature.numpy()

        feature = feature.reshape((-1,))
        return feature

class EfficientNet:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "google/efficientnet-b7"
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model = EfficientNetModel.from_pretrained(model_name).to(self.device)

    def extract(self, image) -> np.ndarray:

        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feature = self.model(**inputs)
        feature = feature.last_hidden_state# Extract the hidden states from the last encoder layer

        # Extract specific features based on your needs
        feature = number_dimention_to_global_average(feature) # Average pooling across all tokens

        if torch.cuda.is_available():
            feature = feature.cpu().numpy()
        else:
            feature = feature.numpy()

        feature = feature.reshape((-1,))
        return feature