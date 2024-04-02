import sys
import streamlit as st
from PIL import Image

try:
    sys.path.insert(0, 'ejfeldman7/which_parent')  # noqa
    from library.model import ResNetWrapper # noqa
except ModuleNotFoundError:
    sys.path.insert(0, '/Users/ejfel/Documents/Github/which_parent')  # noqa
    from library.model import ResNetWrapper # noqa


@st.cache(allow_output_mutation=True)
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
    model = ResNetWrapper(num_classes=2)
    main()
