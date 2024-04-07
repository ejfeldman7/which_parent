import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F


# TODO: from facenet_pytorch import MTCNN, InceptionResnetV1
class ResNetWrapper(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNetWrapper, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(2048, self.num_classes)

    def set_features(self, file1, file2, child_img):
        """
        Extracts features from the images for parents and child,
        and sets them as respective attributes
        """
        self.parent1 = self.extract_features(file1)
        self.parent2 = self.extract_features(file2)
        self.child = self.extract_features(child_img)

    def extract_features(self, img):
        """
        Extracts features from an image and returns a tensor
        """
        img_tensor = F.to_tensor(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        features = self.resnet(img_tensor[:, :3, :, :])
        return features

    def import_resnet(self):
        self.resnet = models.resnet50(pretrained=True)

    def get_similarities(self) -> str:
        """
        Computes the cosine similar of the child image with the parent images.
        Returns which parent has the greatest similarity to the child.
        """
        self.resnet = self.load_resnet()
        similarity_to_1 = float(torch.cosine_similarity(self.child, self.parent1, dim=1))
        similarity_to_2 = float(torch.cosine_similarity(self.child, self.parent2, dim=1))

        if similarity_to_1 > similarity_to_2:
            self.likeness = "parent1"
        else:
            self.likeness = "parent2"
        return (
            f"Child is most similar to {self.likeness}",
            similarity_to_1,
            similarity_to_2,
        )
