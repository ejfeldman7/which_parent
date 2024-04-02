import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F

sys.path.insert(0, "/Users/ejfel/Documents/Github/which_parent")  # noqa
from utils.image_loader import convert_to_png


class ResNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super(ResNetWrapper, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)

    def set_features(self, file1, file2, child_img):
        self.parent1 = self.extract_features(file1)
        self.parent2 = self.extract_features(file2)
        self.child = self.extract_features(child_img)

    def extract_features(self, img_path):
        img = convert_to_png(img_path)
        img_tensor = F.to_tensor(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        features = self.resnet(img_tensor[:, :3, :, :])
        return features

    def get_similarities(self):
        similarity1 = torch.cosine_similarity(self.child, self.parent1, dim=1)
        similarity2 = torch.cosine_similarity(self.child, self.parent2, dim=1)

        # Determine which original picture the comparison is most similar to
        if similarity1 > similarity2:
            self.likeness = "parent1"
            return "Child is most similar to Parent in picture 1"
        else:
            self.likeness = "parent2"
            return "Child is most similar to Parent in picture 2"
