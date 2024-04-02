import sys
import logging

from PIL import Image
import streamlit as st 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    sys.path.insert(0, '/ejfeldman7/which_parent')  # noqa
    from library.model import ResNetWrapper # noqa
except ModuleNotFoundError:
    sys.path.insert(0, '/Users/ejfel/Documents/Github/which_parent')  # noqa
    from library.model import ResNetWrapper # noqa


# @st.cache(allow_output_mutation=True)
@st.cache_resource 
def load_model():
    return ResNetWrapper(num_classes=2)


def main(model):
    st.title("Parent-Child Similarity")
    
    # Upload parent images
    parent1 = st.file_uploader(
        "Upload Parent 1 Image", type=["jpg", "jpeg", "png"]
    )
    parent2 = st.file_uploader(
        "Upload Parent 2 Image", type=["jpg", "jpeg", "png"]
    )

    # Upload child image
    child = st.file_uploader("Upload Child Image", type=["jpg", "jpeg", "png"])

    if parent1 and parent2 and child:
        # Display the uploaded images
        parent1_img = Image.open(parent1)
        logger.info(f"Parent 1 image: {parent1} of type {type(parent1_img)}")
        parent2_img = Image.open(parent2)
        logger.info(f"Parent 2 image: {parent2} of type {type(parent2_img)}")
        child_img = Image.open(child)
        logger.info(f"Child image: {child} of type {type(child_img)}")

        st.image(
            [parent1_img, parent2_img, child_img],
            caption=["Parent 1", "Parent 2", "Child"]
        )

        if st.button("Accept"):
            # Compute similarity scores
            model.set_features(parent1_img, parent2_img, child_img)

            outcome = model.get_similarities()
            st.write(outcome[0])
            st.write(f"Parent 1 similarity: {outcome[1]}")
            st.write(f"Parent 2 similarity: {outcome[2]}")


if __name__ == "__main__":
    model = load_model()
    main(model)
