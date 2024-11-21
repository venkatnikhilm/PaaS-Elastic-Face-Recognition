# PaaS Elastic Face Recognition

## Overview

This project implements an end-to-end serverless pipeline for video processing and face recognition using AWS services. The workflow involves splitting videos into frames, detecting faces in the frames, and identifying the closest match using a pre-trained ResNet model.

The pipeline leverages the following AWS services:
- **AWS Lambda**: For serverless execution of the video splitting and face recognition logic.
- **Amazon S3**: For storing input videos, intermediate frames, and face recognition results.
- **Amazon ECR**: For containerized deployment of the face recognition Lambda function.
- **AWS CloudWatch**: For monitoring logs and debugging.

---

## Workflow

1. **Upload Video**: Videos are uploaded to the input S3 bucket (`<ASU_ID>-input`).
2. **Video Splitting**: 
   - A Lambda function processes the video, extracts the first frame using `ffmpeg`, and uploads the frame to the intermediate bucket (`<ASU_ID>-stage-1`).
   - The function also triggers the face recognition Lambda function asynchronously.
3. **Face Recognition**:
   - The face recognition Lambda function processes the frame, detects faces using `MTCNN`, and computes embeddings using a ResNet model.
   - The function compares the embeddings to those in the `data.pt` file and identifies the closest match.
   - Results are saved as `.txt` files in the output bucket (`<ASU_ID>-output`).

---

## Prerequisites

1. **AWS Services**:
   - An AWS account with permissions for Lambda, S3, ECR, and CloudWatch.
   - Pre-configured S3 buckets:
     - `<ASU_ID>-input` for input videos.
     - `<ASU_ID>-stage-1` for intermediate frames.
     - `<ASU_ID>-output` for recognition results.
     - `<ASU_ID>-embeddings` for storing the `data.pt` file.

2. **Docker**:
   - Installed on your local system for containerizing the face recognition Lambda function.

3. **Dependencies**:
   - `boto3`, `torch`, `torchvision`, `opencv-python-headless`, `facenet-pytorch`.

---

# Project Structure

project/
├── video_splitting_lambda/
│   ├── video_splitting.py        # Lambda function for splitting videos
│   ├── requirements.txt        # Dependencies required for video splitting
│   └── ffmpeg_layer/ (optional)  # Custom ffmpeg layer (if used)
│
├── face_recognition_lambda/
│   ├── face-rec.py                # Lambda function for face recognition
│   ├── requirements.txt          # Dependencies required for face recognition
│   ├── requirements2.txt         # Secondary dependency file (likely for PyTorch)
│   ├── Dockerfile                # Dockerfile for deploying the Lambda function
│   ├── entry.sh                   # Entry point script for the Docker container
│   └── data.pt                    # Pre-trained embeddings file for face recognition
│
├── s3_buckets/
│   ├── input/                    # S3 bucket for storing input videos
│   ├── stage-1/                   # S3 bucket for storing intermediate frames (results of video splitting)
│   └── output/                   # S3 bucket for storing face recognition results
│
└── README.md                     # Project 

## Deployment Steps

### 1. **Video Splitting Lambda**
1. Create a Lambda function in the AWS Management Console.
2. Add a pre-built `ffmpeg` layer or upload a custom ffmpeg layer to the Lambda function.
3. Upload the `video_splitting.py` code and its dependencies.
4. Configure the S3 trigger for the input bucket (`<ASU_ID>-input`).

### 2. **Face Recognition Lambda**
1. Build the Docker image for face recognition:
   ```bash
   docker build -t face-recognition .
   ```
2.	Tag and push the Docker image to Amazon ECR:
  ```bash
  docker tag face-recognition:latest <account_id>.dkr.ecr.<region>.amazonaws.com/face-recognition:latest
  docker push <account_id>.dkr.ecr.<region>.amazonaws.com/face-recognition:latest
  ```
3.	Create a new Lambda function and select “Container Image” as the deployment type.
4.	Use the uploaded image from ECR.
## Testing the Pipeline

1.	Upload a video file to the <ASU_ID>-input bucket.
2.	Monitor the intermediate frames in the <ASU_ID>-stage-1 bucket.
3.	Check for .txt files with predictions in the <ASU_ID>-output bucket.
