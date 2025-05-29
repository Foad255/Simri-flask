
# ml_embedding_service/utils.py
import boto3
from botocore.exceptions import ClientError
import os
import tempfile
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Initialize S3 client. Credentials should be configured in the environment
# (e.g., via AWS CLI config, IAM roles for EC2/ECS, or environment variables)
s3_client = boto3.client(
    's3',
    # region_name=os.getenv("AWS_REGION"), # Optional: if not set by default profile/role
    # aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_ML"), # Avoid hardcoding
    # aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_ML")
)

def download_files_from_s3(bucket_name: str, s3_keys: dict):
    """
    Downloads multiple files from S3 to a single temporary directory.

    Args:
        bucket_name (str): The S3 bucket name.
        s3_keys (dict): A dictionary mapping modality (e.g., 't1c') to its S3 object key.
                        Example: {'t1c': 'patients/id1/t1c.nii.gz', 't1n': 'patients/id1/t1n.nii.gz'}

    Returns:
        tuple: (temp_file_paths, cleanup_callback)
            temp_file_paths (dict): Dictionary mapping modality to its temporary local file path.
            cleanup_callback (function): A function to call to remove the temporary directory and its contents.

    Raises:
        FileNotFoundError: If any S3 object is not found or cannot be accessed.
        ClientError: For other S3 related errors.
    """
    temp_dir = tempfile.mkdtemp(prefix="mri_s3_downloads_")
    logger.info(f"Created temporary directory for S3 downloads: {temp_dir}")

    downloaded_file_paths = {}

    modalities_to_download = list(s3_keys.keys()) # Get all modalities user sent

    for modality_key, s3_object_key in s3_keys.items():
        if not s3_object_key: # Skip if a key is empty or None
            logger.warning(f"No S3 object key provided for modality '{modality_key}', skipping download for this modality.")
            downloaded_file_paths[modality_key] = None # Explicitly mark as not downloaded
            continue

        # Construct a local file name, try to keep original extension if possible
        base_name = os.path.basename(s3_object_key)
        local_file_path = os.path.join(temp_dir, f"{modality_key}_{base_name}") # Prefix with modality to avoid name clashes

        try:
            logger.info(f"Attempting to download s3://{bucket_name}/{s3_object_key} to {local_file_path}")
            s3_client.download_file(bucket_name, s3_object_key, local_file_path)
            downloaded_file_paths[modality_key] = local_file_path
            logger.info(f"Successfully downloaded {s3_object_key} to {local_file_path}")
        except ClientError as e:
            if e.response['Error']['Code'] == '404' or 'NoSuchKey' in str(e):
                logger.error(f"S3 object not found: s3://{bucket_name}/{s3_object_key}. Error: {e}")
                # Store None for this path, embedder.py will handle it by creating zeros
                downloaded_file_paths[modality_key] = None
                # Continue to download other files if possible, or decide to raise immediately
                # For your use case, filling with zeros is the desired behavior for missing files.
            else:
                logger.error(f"S3 ClientError downloading {s3_object_key}: {e}", exc_info=True)
                # For other S3 errors, it might be more critical
                _cleanup_temp_dir(temp_dir) # Attempt cleanup on error
                raise # Re-raise the S3 client error
        except Exception as e:
            logger.error(f"Unexpected error downloading {s3_object_key}: {e}", exc_info=True)
            _cleanup_temp_dir(temp_dir) # Attempt cleanup on error
            raise # Re-raise unexpected error

    def cleanup():
        _cleanup_temp_dir(temp_dir)

    return downloaded_file_paths, cleanup

def _cleanup_temp_dir(dir_path):
    """Safely removes a directory and its contents."""
    try:
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(dir_path)
            logger.info(f"Successfully cleaned up temporary directory: {dir_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory {dir_path}: {e}", exc_info=True)
