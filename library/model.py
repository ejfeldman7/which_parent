import streamlit as st  # noqa
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F
import torch.nn.utils.prune as prune
import torch.quantization


class ResNetWrapper(nn.Module):
    def __init__(self, num_classes: int, prune_model: bool = False):
        super(ResNetWrapper, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(512, self.num_classes)
        if prune_model:
            prune.l1_unstructured(self.fc, name="weight", amount=0.5)

    def quantize_model(self):
        self.resnet = torch.quantization.quantize_dynamic(
            self.resnet, {torch.nn.Linear}, dtype=torch.qint8
        )

    def extract_features(self, img):
        """
        Extracts features from an image and returns a tensor
        """
        img_tensor = F.to_tensor(img)
        unsqueezed_tensor = torch.unsqueeze(img_tensor, 0)
        features = self.resnet(unsqueezed_tensor[:, :3, :, :])
        st.success("Features exptracted and formatted...")
        return features

    def import_resnet(self, weights: str = None):
        if not weights:
            self.resnet = models.resnet18(pretrained=True)
            st.success("Loaded pretrained ResNet from TorchVision!")
        else:
            self.resnet = models.resnet18(weights=None if weights == "Random" else weights)
            st.success(f"Loaded ResNet from TorchVision with {weights} weights!")

    def get_similarities(self, child, parent1, parent2) -> str:
        """
        Computes the cosine similar of the child image with the parent images.
        Returns which parent has the greatest similarity to the child.
        """
        similarity_to_1 = float(
            torch.cosine_similarity(child, parent1, dim=1)
        )
        similarity_to_2 = float(
            torch.cosine_similarity(child, parent2, dim=1)
        )

        if similarity_to_1 > similarity_to_2:
            likeness = "parent 1"
        else:
            likeness = "parent 2"
        return (
            f"Based on these images, this child is most similar to {likeness}",
            similarity_to_1,
            similarity_to_2,
        )


class FamilyValues():
    def __init__(self, parent1_img, parent2_img, child_img, resnet_wrapper: ResNetWrapper):
        self.parent1_img = parent1_img
        self.parent2_img = parent2_img
        self.child_img = child_img
        self.parent1 = resnet_wrapper.extract_features(self.parent1_img)
        self.parent2 = resnet_wrapper.extract_features(self.parent2_img)
        self.child = resnet_wrapper.extract_features(self.child_img)
