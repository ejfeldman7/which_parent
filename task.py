import sys
import logging

from PIL import Image
import streamlit as st # noqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    sys.path.insert(0, '/ejfeldman7/which_parent')  # noqa
    from library.model import ResNetWrapper # noqa
except ModuleNotFoundError:
    sys.path.insert(0, '/Users/ejfel/Documents/Github/which_parent')  # noqa
    from library.model import ResNetWrapper # noqa


@st.cache_resource(experimental_allow_widgets=True)
def load_model():
    model = ResNetWrapper(num_classes=2)
    weights = st.checkbox("Check to select a different set of weights for ResNet than pretrained:")
    if weights:
        option = st.selectbox("What weights would you like to use for ResNet?",
        ("DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2", "Random", ))
    model.import_resnet(option if option else None)
    return model


def main():
    st.sidebar.write(
    '''
    __About__ \n
    This project was built to help settle those questions about who a child looks more like.  \r\n
    The MVP implementation wraps Resnet50 to produce tensors for all three images.  \r\n
    Those tensors of the parent images are then compared to determine which are "closest" to child. \r\n
    '''
    '''
    \r\n
    This site was created by Ethan Feldman. You can find him on [GitHub](https://github.com/ejfeldman7), [LinkedIn](https://www.linkedin.com/in/feldmanethan/),
    [Medium/TDS](https://ethan-feldman.medium.com/) and on his [website](https://www.ejfeldman.com/)  \r\n
    ''')

    st.title("Which-Parent")
    '''
    The Streamlit App using computer vision to answer life's big questions.
    '''
    st.write("Upload two parent images and one child image to determine which parent the child looks more like.")
    
    # Load the model
    model = load_model()
    
    # Upload parent images
    parent1 = st.file_uploader(
        "Upload Parent 1 Image", type=["jpg", "jpeg", "png"]
    )
    parent2 = st.file_uploader(
        "Upload Parent 2 Image", type=["jpg", "jpeg", "png"]
    )
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

        if st.button("Click Accept to Run"):
            model.set_features(parent1_img, parent2_img, child_img)

            outcome = model.get_similarities()
            st.write(outcome[0])
            st.write(f"Parent 1 similarity: {outcome[1]}")
            st.write(f"Parent 2 similarity: {outcome[2]}")


if __name__ == "__main__":
    main()
