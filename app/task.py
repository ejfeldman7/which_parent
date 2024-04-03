import sys
import streamlit as st
from PIL import Image

try:
    # sys.path.insert(0, '/ejfeldman7/which_parent')  # noqa
    from library.model import ResNetWrapper # noqa
except ModuleNotFoundError:
    sys.path.insert(0, '/Users/ejfel/Documents/Github/which_parent')  # noqa
    from library.model import ResNetWrapper # noqa


@st.cache(allow_output_mutation=True)
def main():
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

        st.button("Click HERE to Run")

        # Compute similarity scores
        ResNetWrapper.set_features(parent1_img, parent2_img, child_img)

        outcome = model.get_similarities()

        # Determine the most similar parent
        st.write(outcome[0])
        st.write(f"Parent 1 similarity: {outcome[1]}")
        st.write(f"Parent 2 similarity: {outcome[2]}")

        '''
        \r\n
        This site was created by Ethan Feldman. You can find him on [GitHub](https://github.com/ejfeldman7), [LinkedIn](https://www.linkedin.com/in/feldmanethan/),
        [Medium/TDS](https://ethan-feldman.medium.com/) and on his [website](https://www.ejfeldman.com/)  \r\n
        '''


if __name__ == "__main__":
    model = ResNetWrapper(num_classes=2)
    main()
