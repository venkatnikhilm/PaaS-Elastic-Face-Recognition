import boto3
import subprocess
import os
import json
import logging

# Initialize AWS clients
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        # Retrieve bucket name and video key from the event
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        video_key = event['Records'][0]['s3']['object']['key']
        
        # Define paths and file names
        local_video_path = f'/tmp/{video_key}'
        video_basename = os.path.splitext(video_key)[0]
        output_frame_path = f'/tmp/{video_basename}.jpg'
        stage1_bucket = '1228911985-stage-1'  # Update with your stage-1 bucket name
        output_frame_key = f"{video_basename}.jpg"  # Frame filename
        
        # Download the video from S3
        s3_client.download_file(bucket_name, video_key, local_video_path)
        
        # Run ffmpeg to extract a single frame
        ffmpeg_command = f"ffmpeg -i {local_video_path} -vframes 1 -vf scale=640:480 {output_frame_path} -y"
        subprocess.run(ffmpeg_command, shell=True, check=True)

        s3_client.upload_file(output_frame_path, stage1_bucket, output_frame_key)
        
        payload = {
            "bucket_name": stage1_bucket,
            "img_file_name": output_frame_key
        }
        
        # Log the payload
        logger.info(f"Payload sent to face-recognition Lambda: {json.dumps(payload)}")
        
        # Invoke the face-recognition Lambda function asynchronously
        lambda_client.invoke(
            FunctionName="face-recognition",  
            InvocationType="Event",
            Payload=json.dumps(payload)
        )
        
        # Clean up temporary files
        os.remove(local_video_path)
        os.remove(output_frame_path)
        
        return {
            'statusCode': 200,
            'body': f"Frame extracted and uploaded to {stage1_bucket}/{output_frame_key}"
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg error: {e}")
        return {
            'statusCode': 500,
            'body': "Error during frame extraction."
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'body': f"Unexpected error: {str(e)}"
        }