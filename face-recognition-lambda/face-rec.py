#__copyright__   = "Copyright 2024, VISA Lab"
#__license__     = "MIT"

import os
import boto3
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Initialize AWS S3 client
s3 = boto3.client("s3")

# Define directories and bucket names
out_dir_path = "/tmp/"
stage1_bucket = "1228911985-stage-1" 
output_bucket = "1228911985-output"  
data_path = "./data.pt" 

# Ensure facenet_pytorch uses /tmp for storing model weights
os.environ["TORCH_HOME"] = "/tmp"

# Initialize MTCNN and ResNet
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained="vggface2").eval()

def download_file_from_s3(bucket_name, s3_object_key, local_file_path):
    try:
        s3.download_file(bucket_name, s3_object_key, local_file_path)
        print(f"Downloaded {s3_object_key} from bucket {bucket_name} to {local_file_path}")
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False

def upload_file_to_s3(bucket_name, output_file):
    try:
        s3.upload_file(out_dir_path + output_file, bucket_name, output_file)
        print(f"Uploaded {output_file} to bucket {bucket_name}")
    except Exception as e:
        print(f"Error uploading to S3: {e}")

def face_recognition_function(key_path):
    try:
        # Read the image using OpenCV
        img = cv2.imread(key_path, cv2.IMREAD_COLOR)
        
        # Detect faces in the image
        boxes, _ = mtcnn.detect(img)
        if boxes is None:
            print("No face detected in the image.")
            return None
        
        # Convert the image to PIL format for further processing
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        face, prob = mtcnn(img, return_prob=True, save_path=None)
        if face is None:
            print("No face detected in the image.")
            return None
        
        # Load embeddings and names from the pre-trained data file
        saved_data = torch.load(data_path)
        embedding_list = saved_data[0]  
        name_list = saved_data[1]  
        
        # Compute the embedding for the detected face
        emb = resnet(face.unsqueeze(0)).detach()
        distances = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
        
        # Find the closest match
        idx_min = distances.index(min(distances))
        return name_list[idx_min]
    except Exception as e:
        print(f"Error in face recognition: {e}")
        return None

def handler(event, context):
    try:
        # Extract file name from the event
        s3_object_key = event["img_file_name"]
        local_file_path = os.path.join(out_dir_path, s3_object_key)
        
        # Download the image from the stage-1 bucket
        download_success = download_file_from_s3(stage1_bucket, s3_object_key, local_file_path)
        if not download_success:
            return {
                "statusCode": 500,
                "body": f"Failed to download {s3_object_key} from bucket {stage1_bucket}."
            }
        
        # Perform face recognition
        recognized_name = face_recognition_function(local_file_path)
        if not recognized_name:
            return {
                "statusCode": 400,
                "body": f"No face detected or recognized in {s3_object_key}."
            }
        
        # Save the recognized name to a .txt file
        output_file = s3_object_key.split(".")[0] + ".txt"
        with open(out_dir_path + output_file, "w") as f:
            f.write(recognized_name)
        
        # Upload the result to the output bucket
        upload_file_to_s3(output_bucket, output_file)
        
        return {
            "statusCode": 200,
            "body": f"Recognized face: {recognized_name}"
        }
    except Exception as e:
        print(f"Error in handler: {e}")
        return {
            "statusCode": 500,
            "body": f"Error: {e}"
        }