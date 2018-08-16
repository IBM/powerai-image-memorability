# Predicting Image Memorability with NASNet in Keras on PowerAI

This code pattern will enable you to build an application that predicts how "unique" or "memorable" images are. You'll do this through the Keras deep learning library, using the NASNet architecture. The dataset this neural network will be trained on is called "LaMem" (Large-scale Image Memorability), by MIT. In order to process the 45,000 training images and 10,000 testing images (224x224 RGB) efficiently, we'll be training the neural network on a PowerAI machine on NIMBIX, enabling us to benefit from NVLink (direct CPU-GPU memory interconnect) without needing any extra code.

## Flow

TODO: add flow diagram

1. A Keras model is trained with the LaMem dataset.
1. The Keras model is converted to a CoreML model.
1. The user uploads their image to the kitura web app.
1. The Kitura web app uses the CoreML model for predictions.
1. The user recieves the neural network's prediction.

## Included Components

* [IBM Power Systems](https://www-03.ibm.com/systems/power/): A server built with open technologies and designed for mission-critical applications.
* [IBM PowerAI](https://www.ibm.com/ms-en/marketplace/deep-learning-platform): A software platform that makes deep learning, machine learning, and AI more accessible and better performing.
* [Kitura](https://www.kitura.io): Kitura is a free and open-source web framework written in Swift, developed by IBM and licensed under Apache 2.0. It’s an HTTP server and web framework for writing Swift server applications.

## Featured Technologies

* [Artificial Intelligence](https://medium.com/ibm-data-science-experience): Artificial intelligence can be applied to disparate solution spaces to deliver disruptive technologies.
* [Swift on the Server](https://developer.ibm.com/swift/): Build powerful, fast and secure server side Swift apps for the Cloud.

# Prerequisites

* If you don't already have a PowerAI server, you can acquire one from [Nimbix](https://www.nimbix.net/ibm) or from the [PowerAI offering](https://console.bluemix.net/catalog/services/powerai) on IBM Cloud.

# Steps

1. [Clone the repo](#1-clone-the-repo)
1. [Download the LaMem dataset](#2-download-the-lamem-dataset)
1. [Train the Keras model](#3-train-the-keras-model)
1. [Convert the Keras model to a CoreML model](#4-convert-the-keras-model-to-a-coreml-model)
1. [Run the Kitura web app](#5-run-the-kitura-web-app)

### 1. Clone the repo

Clone the `powerai-image-memorability` repo locally. In a terminal, run:

```
git clone https://www.github.com/IBM/powerai-image-memorability
```

### 2. Download the LaMem data

To simplify the data download and extraction procedure, this code pattern contains a script that you can run to download and extract the data for you. To run this script, `cd` into the `powerai_serverside` directory in a terminal:

```
cd powerai_serverside
```

Then, run the following command to execute the script:

```
sh download_extract_data.sh
```

### 3. Train the Keras model

To train the keras model
