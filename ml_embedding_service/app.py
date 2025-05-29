# ml_embedding_service/app.py
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import logging

from embedder import get_mri_embedding # Core logic will be in embedder.py
from utils import download_files_from_s3 # S3 download helper

# Load environment variables from .env file (optional, good for local dev)
load_dotenv()

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
# These can also be loaded from environment variables if preferred
# S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME") # If the bucket is fixed for the service
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Handled in embedder.py

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logger.info("Health check endpoint called.")
    return jsonify({"status": "healthy"}), 200

@app.route('/embed', methods=['POST'])
def embed_mri_data():
    """
    Endpoint to generate MRI embeddings.
    Expects JSON payload:
    {
      "patient_id": "string",
      "s3_bucket": "string", // Bucket where the files are stored
      "s3_keys": {
        "t1c": "path/to/t1c.nii.gz",
        "t1n": "path/to/t1n.nii.gz",
        "t2f": "path/to/t2f.nii.gz",
        "t2w": "path/to/t2w.nii.gz"
        // seg is not used for embedding in your script, so not listed here
      }
    }
    Returns:
    {
      "patient_id": "string",
      "embedding": [float, float, ..., float] // 128-dim vector
    }
    Or an error JSON.
    """
    logger.info("Received request for /embed")
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON payload received.")
            return jsonify({"error": "No JSON payload received"}), 400

        patient_id = data.get('patient_id')
        s3_bucket = data.get('s3_bucket')
        s3_keys = data.get('s3_keys') # This is a dict of modality -> s3_key

        if not all([patient_id, s3_bucket, s3_keys]):
            logger.error(f"Missing required fields. Received: patient_id={patient_id}, s3_bucket={s3_bucket}, s3_keys={s3_keys is not None}")
            return jsonify({"error": "Missing required fields: patient_id, s3_bucket, s3_keys"}), 400

        if not isinstance(s3_keys, dict):
            logger.error("s3_keys must be a dictionary.")
            return jsonify({"error": "s3_keys must be a dictionary (modality: key)"}), 400

        logger.info(f"Processing embedding for patient_id: {patient_id}, bucket: {s3_bucket}")

        # 1. Download files from S3 to a temporary location
        # The s3_keys dict should map modality (e.g., 't1c') to its S3 object key
        # The download_files_from_s3 function will return a dict of modality -> local_temp_path
        temp_file_paths, cleanup_callback = download_files_from_s3(s3_bucket, s3_keys)
        logger.info(f"Files downloaded to temporary paths: {temp_file_paths}")

        # 2. Generate embedding using the downloaded files
        # The get_mri_embedding function will use these local paths
        embedding_vector = get_mri_embedding(temp_file_paths, patient_id) # patient_id for logging inside

        # 3. Cleanup temporary files
        try:
            cleanup_callback()
            logger.info(f"Temporary files cleaned up for patient {patient_id}")
        except Exception as e:
            logger.error(f"Error during temporary file cleanup for patient {patient_id}: {e}")
            # Non-critical for the response, but should be logged

        logger.info(f"Successfully generated embedding for patient_id: {patient_id}")
        return jsonify({
            "patient_id": patient_id,
            "embedding": embedding_vector # Should be a list of floats
        }), 200

    except FileNotFoundError as e: # Specific error from S3 download or local file access
        logger.error(f"File not found error during embedding for patient {data.get('patient_id', 'N/A')}: {e}")
        # Cleanup if partially downloaded
        if 'cleanup_callback' in locals() and callable(cleanup_callback):
            try: cleanup_callback()
            except: pass
        return jsonify({"error": f"File processing error: {str(e)}"}), 404
    except ValueError as e: # Specific error for invalid inputs to model/transforms
        logger.error(f"Value error during embedding for patient {data.get('patient_id', 'N/A')}: {e}")
        if 'cleanup_callback' in locals() and callable(cleanup_callback):
            try: cleanup_callback()
            except: pass
        return jsonify({"error": f"Input data error: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Unexpected error during embedding for patient {data.get('patient_id', 'N/A')}: {e}", exc_info=True)
        # Ensure cleanup is attempted even on unexpected errors
        if 'cleanup_callback' in locals() and callable(cleanup_callback):
            try: cleanup_callback()
            except: pass
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # For development, you can run this directly.
    # For production, use a WSGI server like Gunicorn or uWSGI.
    # Example: gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")
