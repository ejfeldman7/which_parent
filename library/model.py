import streamlit as st  # noqa
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F
import torch.nn.utils.prune as prune
import torch.quantization


# TODO: from facenet_pytorch import MTCNN, InceptionResnetV1
class ResNetWrapper(nn.Module):
    def __init__(self, num_classes: int, prune_model: bool = False):
        super(ResNetWrapper, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(2048, self.num_classes)
        if prune_model:
            prune.l1_unstructured(self.fc, name="weight", amount=0.5)

    def quantize_model(self):
        self.resnet = torch.quantization.quantize_dynamic(
            self.resnet, {torch.nn.Linear}, dtype=torch.qint8
        )

    def set_features(self, file1, file2, child_img):
        """
        Extracts features from the images for parents and child,
        and sets them as respective attributes
        """
        st.write("Generating features for parent1...")
        self.parent1 = self.extract_features(file1)
        st.write("Generating features for parent2...")
        self.parent2 = self.extract_features(file2)
        st.write("Generating features for child...")
        self.child = self.extract_features(child_img)

    def extract_features(self, img):
        """
        Extracts features from an image and returns a tensor
        """
        img_tensor = F.to_tensor(img)
        st.success("to tensors...")
        unsqueezed_tensor = torch.unsqueeze(img_tensor, 0)
        st.success("unsqueezed...")
        features = self.resnet(unsqueezed_tensor[:, :3, :, :])
        st.success("Features exptracted and formatted...")
        return features

    def import_resnet(self, weights: str = None):
        if not weights:
            self.resnet = models.resnet50(pretrained=True)
            st.success("Loaded pretrained ResNet50 from TorchVision!")
        else:
            self.resnet = models.resnet50(weights=None if weights == "Random" else weights)
            st.success(f"Loaded ResNet50 from TorchVision with {weights} weights!")

    def get_similarities(self) -> str:
        """
        Computes the cosine similar of the child image with the parent images.
        Returns which parent has the greatest similarity to the child.
        """
        similarity_to_1 = float(
            torch.cosine_similarity(self.child, self.parent1, dim=1)
        )
        similarity_to_2 = float(
            torch.cosine_similarity(self.child, self.parent2, dim=1)
        )

        if similarity_to_1 > similarity_to_2:
            self.likeness = "parent1"
        else:
            self.likeness = "parent2"
        return (
            f"Child is most similar to {self.likeness}",
            similarity_to_1,
            similarity_to_2,
        )
