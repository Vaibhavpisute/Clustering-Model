# Clustering-Model #image segmentation
This project applies K-Means clustering to segment images based on pixel color similarity. It includes a simple Tkinter GUI for selecting images and specifying cluster count, then visualizes and saves the clustered output. Ideal for experimentation with unsupervised learning and image segmentation.
# K-Means Image Clustering Tool

This is a Python-based GUI tool for applying **K-Means clustering** to images. It allows users to select an image from a dataset and cluster its pixels into a specified number of groups (colors), helping in segmentation or artistic transformation of images.

## Features

- GUI file selection for both image folders and files (using `tkinter`)
- Normalize and denormalize image colors (with `MinMaxScaler`)
- Apply K-Means clustering to image pixels (`scikit-learn`)
- Save and display both original and clustered versions of the image

## Requirements

Install the following Python libraries before running:

Donwload the dataset file placed in a directory , copy that directory and paste inside the code

```bash
pip install numpy matplotlib scikit-learn scikit-image
