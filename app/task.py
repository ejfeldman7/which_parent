import streamlit as st
from PIL import Image
from torchvision.transforms import ToTensor

sys.path.insert(0, '/Users/ejfel/Documents/Github/which_parent')  # noqa
from library.model import ResNetWrapper

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
model = ResNetWrapper(num_classes=2)

    # parent1_tensor = ToTensor()(parent1).unsqueeze(0)
    # parent2_tensor = ToTensor()(parent2).unsqueeze(0)
    # child_tensor = ToTensor()(child).unsqueeze(0)

    # # Compute the similarity scores
    # similarity_parent1 = model.compute_similarity(parent1_tensor, child_tensor)
    # similarity_parent2 = model.compute_similarity(parent2_tensor, child_tensor)

# Streamlit app
def main():
    st.title("Parent-Child Similarity")

    # Upload parent images
    parent1 = st.file_uploader("Upload Parent 1 Image", type=["jpg", "jpeg", "png"])
    parent2 = st.file_uploader("Upload Parent 2 Image", type=["jpg", "jpeg", "png"])

    # Upload child image
    child = st.file_uploader("Upload Child Image", type=["jpg", "jpeg", "png"])

    if parent1 and parent2 and child:
        # Display the uploaded images
        parent1_img = Image.open(parent1)
        parent2_img = Image.open(parent2)
        child_img = Image.open(child)

        st.image([parent1_img, parent2_img, child_img], caption=["Parent 1", "Parent 2", "Child"])

        # Compute similarity scores
        ResNetWrapper.set_features(parent1_img, parent2_img, child_img)

        outcome = model.get_similarities()

        # Determine the most similar parent
        st.write(outcome)

if __name__ == "__main__":
    main()