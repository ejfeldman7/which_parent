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
    st.sidebar.write(
    '''
    __About__ \n
    This project was built from just under 6000 reviews from www.coffeereview.com. The blind reviews were used to create nine-dimensional flavor vectors for comparisons between coffees.
    \n
    This site was created by Ethan Feldman. You can find him on [GitHub](https://github.com/ejfeldman7), [LinkedIn](https://www.linkedin.com/in/feldmanethan/), [Medium/TDS](https://ethan-feldman.medium.com/) and eventually on his website (link to come)!
    ''')

    st.title("Parent-Child Similarity")
    '''
    This project was built to help settle those questions about who a child looks more like.  \r\n
    The MVP implementation wraps Resnet50 to produce tensors for all three images.  \r\n
    Those tensors of the parent images are then compared to determine which are "closest" to child. \r\n
    '''
    st.write("Upload two parent images and one child image to determine which parent the child looks more like.")
    # Upload parent images
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
            st.write("<div style='text-align: center;'>Accept</div>", unsafe_allow_html=True)
            # Compute similarity scores
            model.set_features(parent1_img, parent2_img, child_img)

            outcome = model.get_similarities()
            st.write(outcome[0])
            st.write(f"Parent 1 similarity: {outcome[1]}")
            st.write(f"Parent 2 similarity: {outcome[2]}")
    
     '''
    \r\n
    This site was created by Ethan Feldman. You can find him on [GitHub](https://github.com/ejfeldman7), [LinkedIn](https://www.linkedin.com/in/feldmanethan/),
    [Medium/TDS](https://ethan-feldman.medium.com/) and on his [website](https://www.ejfeldman.com/)  \r\n
    '''


if __name__ == "__main__":
    model = load_model()
    main(model)
