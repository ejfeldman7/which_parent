import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F

try:
    sys.path.insert(0, '/ejfeldman7/which_parent')  # noqa
    from utils.image_loader import ResNetWrapper # noqa
except ModuleNotFoundError:
    sys.path.insert(0, '/Users/ejfel/Documents/Github/which_parent')  # noqa
    from utils.image_loader import ResNetWrapper # noqa


class ResNetWrapper(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNetWrapper, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)

    def set_features(self, file1: str, file2: str, child_img: str):
        '''
        Extracts features from the images for parents and child,
        and sets them as respective attributes
        '''
        self.parent1 = self.extract_features(file1)
        self.parent2 = self.extract_features(file2)
        self.child = self.extract_features(child_img)

    def extract_features(self, img_path: str) -> torch.Tensor:
        '''
        Extracts features from an image and returns a tensor
        '''
        img = convert_to_png(img_path)
        img_tensor = F.to_tensor(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        features = self.resnet(img_tensor[:, :3, :, :])
        return features

    def get_similarities(self) -> str:
        '''
        Computes the cosine similar of the child image with the parent images.
        Returns which parent has the greatest similarity to the child.
        '''
        similarity1 = torch.cosine_similarity(self.child, self.parent1, dim=1)
        similarity2 = torch.cosine_similarity(self.child, self.parent2, dim=1)

        # Determine which original picture the comparison is most similar to
        if similarity1 > similarity2:
            self.likeness = "parent1"
            return "Child is most similar to Parent in picture 1"
        else:
            self.likeness = "parent2"
            return "Child is most similar to Parent in picture 2"
