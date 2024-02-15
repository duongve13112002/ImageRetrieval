# Image Retrieval Web Application

## Overview

Welcome to the Image Retrieval Web Application repository! This project focuses on creating a web-based image retrieval system using Gradio library for the frontend and a combination of Convolutional Neural Networks (CNNs) and Transformers from the Hugging Face Transformers library for feature extraction.

## Features

- **Web Interface**: The application provides a user-friendly web interface powered by Gradio, allowing users to easily interact with the image retrieval system.

- **Feature Extraction**: The backend utilizes state-of-the-art CNNs and Transformers to extract meaningful feature vectors from images. This enables efficient and accurate image retrieval.

- **Model Variety**: The project supports various CNN architectures and Transformers, allowing users to choose the best model for their specific needs.

## Models

* In this project, we have implemented 6 models for experimentation. They are as follows:
  * VisionTransformer
  * BEiT
  * MobileViTV2
  * Bit
  * EfficientFormer
  * MobileNetV2
  * ResNet
  * EfficientNet

## Datasets
### Paris Buildings

- The "Paris Buildings" dataset comprises a collection of high-resolution images capturing various architectural structures across Paris. The dataset is carefully annotated, providing information on building types, architectural styles, and geographic locations. Researchers and developers can leverage this dataset for tasks such as image classification, object detection, and scene understanding.

- For this dataset, it is divided into two parts. You can download them here:
   * [paris_part1](https://thor.robots.ox.ac.uk/datasets/paris-buildings/paris_1-v1.tgz).
   * [paris_part2](https://thor.robots.ox.ac.uk/datasets/paris-buildings/paris_2-v1.tgz).

- The groundtruth can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz).

### Oxford Buildings

- The "Oxford Buildings" dataset focuses on architectural diversity within the city of Oxford. Similar to the Paris dataset, it includes annotated images to facilitate research in computer vision and related fields. This dataset is particularly suitable for tasks involving cross-city analysis or comparative studies between different architectural environments.

- The images in the Oxford Buildings dataset can be found [here](https://thor.robots.ox.ac.uk/datasets/oxford-buildings/oxbuild_images-v1.tgz).
- The groundtruth can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz).

## Feature vector comparison algorithm

### Euclidean Distance

The Euclidean distance measures the straight-line distance between two points in Euclidean space. For vectors $\(X\)$ and \(Y\) of dimension $\(n\)$, the Euclidean distance ($\(ED\)$) is calculated as follows:

$\[ ED(X, Y) = \sqrt{\sum_{i=1}^{n} (X_i - Y_i)^2} \]$

This method is suitable for scenarios where the absolute magnitude and direction of the vectors are crucial for similarity assessment.

### Cosine Similarity

Cosine similarity calculates the cosine of the angle between two vectors in multi-dimensional space. For vectors \(X\) and \(Y\) of dimension \(n\), the cosine similarity (\(CS\)) is computed as:

\[ CS(X, Y) = \frac{\sum_{i=1}^{n} (X_i \times Y_i)}{\sqrt{\sum_{i=1}^{n} (X_i)^2} \times \sqrt{\sum_{i=1}^{n} (Y_i)^2}} \]

This method is effective when the direction of vectors is more important than their magnitudes. It is widely used in text mining, document analysis, and recommendation systems.

### Time Series Similarity (TS_SS)

Time Series Similarity (TS_SS) is a method specifically designed for comparing the similarity between two time series. It takes into account both amplitude and phase information, making it suitable for time-dependent data. The TS_SS similarity (\(TS\_SS\)) is computed using a dynamic programming approach, providing robust results even in the presence of temporal misalignments.

\[ TS\_SS(X, Y) = \frac{\sum_{i=1}^{n} (X_i - Y_i)^2}{\sqrt{\sum_{i=1}^{n} (X_i)^2 + \sum_{i=1}^{n} (Y_i)^2}} \]

This method is particularly useful for applications involving time series data, such as signal processing, financial analysis, and environmental monitoring.

## Video Demo

For a visual representation of the project in action, check out the [Video Demo](demo/Video-demo.mp4).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/duongve13112002/ImageRetrieval.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook (Detailed instructions on how to run are provided in here):
   ```bash
   jupyter notebook notebook.ipynb
   ```

   Visit `http://localhost:7860` in your web browser to access the Image Retrieval Web Application.


## Usage

1. Upload Image: Use the web interface to upload an image for retrieval.

2. Retrieve Similar Images: The system will extract features using the selected model and display similar images from the dataset.

## Contributing

Contributions are welcome! If you find any issues or have ideas for improvements, please create a GitHub issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Gradio: [Gradio Documentation](https://gradio.app/docs)
- Hugging Face Transformers: [Transformers Documentation](https://huggingface.co/transformers/)

Thank you for using and contributing to the Image Retrieval Web Application!
