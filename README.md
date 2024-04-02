# Which-Parent: <br/> Computer Vision Web Applet

A web applet where you can upload the pictures of parents and their child to determine who the kid looks most like, once and for all.

[The Streamlit app can be found here](https://which-parent.streamlit.app/)

# Contents

## In the Notebooks folder you will find:

- library: contains the model.py file containing our model class
- notebooks: notebooks used for exploration and testing
- test_files: test file images used for comparisons
- utils: loading utils and more
- visuals: tbd
- requirements.txt to pin versions of critical packages
- task.py: applet to run on Streamit


# Summary of Work and Findings  

## Process

This project is built on top a wrapped Resnet model, making similarity scores using cosine similarity to compare the Tensors created by images.

