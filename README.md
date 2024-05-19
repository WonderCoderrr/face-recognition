# Face Recognition Project

## Overview

This project is a facial recognition system that detects faces in a video stream and saves the detected face data to a JSON file. It utilizes the OpenCV library for face detection and manipulation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/WonderCoderrr/face-recognition.git
   ```

2. Navigate to the project directory:

   ```bash
   cd face-recognition
   ```

3. Run the setup script to create a new Conda environment, install required packages, and execute the main script:

   ```bash
   source setup.sh 
   ```

## Configuration

The project configuration is stored in the `conf.yaml` file. You can customize the following settings:

- **Input Paths:** Specify the paths to the input video file and the cascade classifier model.
- **Output Paths:** Define the paths for the processed video output and the face data JSON output.
- **Model Parameters:** Set the parameters for the face detection model, such as scale factor, minimum neighbors, minimum size, and maximum size.
- **Process Options:** Toggle options for saving the processed video and the face data JSON file.
- **Video Input Example:** You can download an example video for testing from [here](https://drive.google.com/file/d/1aHuw1rwIvBxyMieKHXsxKdbbWKBioqiU/view). Place the downloaded video in the `videos` folder.
- **Video Output Example:** You can download an example of video result from [here](https://drive.google.com/file/d/15L90lZd4uUIQdHt1Zx9eu8fGzS04KZrz/view?usp=sharing). 

## Dependencies

- Python 3.10
- OpenCV
- PyYAML

## Usage

Once the setup is complete, you can run the main Python script to process the video:

```bash
python main.py
```

The processed video with face detection annotations and the face data JSON file will be saved in the specified output paths.


## Pre-Trained Model License

The pre-trained model `haarcascade_frontalface_default.xml` used in this project is provided by OpenCV. Please refer to the [OpenCV GitHub repository](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) for the license information.

For the most accurate and up-to-date information regarding the license, refer directly to the LICENSE file in the OpenCV repository.
