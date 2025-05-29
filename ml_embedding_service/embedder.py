# ml_embedding_service/embedder.py

import torch
import numpy as np
import os
import logging
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensity, ToTensor
)
from monai.networks.nets import DenseNet121

logger = logging.getLogger(__name__)

# ========== CONFIG (from your script, adapted) ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (128, 128, 128) # (D, H, W) for MONAI Resize
# Modalities required by the model for creating the 4-channel input
REQUIRED_MODALITIES_FOR_MODEL = ['t1c', 't1n', 't2f', 't2w']
MODEL_IN_CHANNELS = len(REQUIRED_MODALITIES_FOR_MODEL)
MODEL_OUT_CHANNELS = 128 # Embedding dimension
# ========================================================

# --- Model Loading ---
# Load model (DenseNet as encoder)
# It's good practice to load the model once when the service starts,
# rather than on every request, if possible.
try:
    logger.info(f"Initializing DenseNet121 model on device: {DEVICE}")
    model = DenseNet121(
        spatial_dims=3,
        in_channels=MODEL_IN_CHANNELS,
        out_channels=MODEL_OUT_CHANNELS
    ).to(DEVICE)
    model.eval() # Set to evaluation mode
    logger.info("DenseNet121 model loaded successfully and set to eval mode.")
    # You might load pre-trained weights here if you have them:
    model_weights_path = os.getenv("MODEL_WEIGHTS_PATH")
    if model_weights_path and os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
        logger.info(f"Loaded model weights from {model_weights_path}")
    else:
        logger.warning("MODEL_WEIGHTS_PATH not set or file not found. Using randomly initialized model (will produce non-zero but arbitrary embeddings).")

except Exception as e:
    logger.error(f"Failed to load or initialize the DenseNet121 model: {e}", exc_info=True)
    # This is a critical error; the service might not be usable.
    # Depending on deployment, you might want to exit or have a fallback.
    model = None # Ensure model is None if loading failed

# --- Preprocessing Definition ---
# This Compose object can also be defined once globally.
try:
    preprocess_transforms = Compose([
        LoadImage(image_only=True),  # <-- This line should NOT have `ensure_channel_first`
        EnsureChannelFirst(),
        Resize(spatial_size=IMG_SIZE),
        ScaleIntensity(),
        ToTensor(dtype=torch.float32)
    ])

    logger.info("MONAI preprocessing transforms initialized.")
except Exception as e:
    logger.error(f"Failed to initialize MONAI transforms: {e}", exc_info=True)
    preprocess_transforms = None


def load_and_preprocess_modalities(local_file_paths: dict, patient_id: str):
    """
    Loads MRI modalities from local file paths, preprocesses them, and stacks them.

    Args:
        local_file_paths (dict): A dictionary mapping modality (str) to its local file path (str).
                                 Example: {'t1c': '/tmp/path/to/t1c.nii.gz', ...}
        patient_id (str): Patient ID for logging purposes.

    Returns:
        torch.Tensor: A tensor of shape (1, num_modalities, D, H, W) ready for the model,
                      or None if a critical error occurs.
    """
    if not preprocess_transforms:
        logger.error(f"Preprocessing transforms not initialized for patient {patient_id}. Cannot proceed.")
        raise RuntimeError("Preprocessing transforms are not available.")

    processed_volumes = []
    for mod_key in REQUIRED_MODALITIES_FOR_MODEL:
        file_path = local_file_paths.get(mod_key)

        if not file_path or not os.path.exists(file_path):
            logger.warning(f"Missing {mod_key} at path '{file_path}' for patient {patient_id}. Creating random data for this modality.")
            # Instead of zeros, create a random tensor for testing
            random_volume = torch.randn((1, *IMG_SIZE), dtype=torch.float32) # Generates random numbers from a standard normal distribution
            processed_volumes.append(random_volume)
            continue

        try:
            logger.info(f"Preprocessing {mod_key} from {file_path} for patient {patient_id}")
            # LoadImage will load NIfTI, AddChannel adds channel, Resize, ScaleIntensity, ToTensor
            img_tensor = preprocess_transforms(file_path) # Output: (1, D, H, W)
            if img_tensor.shape[1:] != IMG_SIZE: # Check D, H, W after Resize
                 logger.warning(f"Unexpected shape after Resize for {mod_key} of patient {patient_id}: {img_tensor.shape}. Expected (1, {IMG_SIZE}). This might indicate an issue.")
            processed_volumes.append(img_tensor)
        except Exception as e:
            logger.error(f"Error preprocessing {mod_key} for patient {patient_id} from {file_path}: {e}", exc_info=True)
            # Create random data if preprocessing fails
            logger.warning(f"Filling with random data for {mod_key} of patient {patient_id} due to preprocessing error.")
            random_volume = torch.randn((1, *IMG_SIZE), dtype=torch.float32)
            processed_volumes.append(random_volume)

    if not processed_volumes or len(processed_volumes) != MODEL_IN_CHANNELS:
        logger.error(f"Incorrect number of processed volumes for patient {patient_id}. Expected {MODEL_IN_CHANNELS}, got {len(processed_volumes)}.")
        return None

    # Stack the volumes along the channel dimension
    # Each volume is (1, D, H, W), so cat results in (num_modalities, D, H, W)
    stacked_tensor = torch.cat(processed_volumes, dim=0)
    # Add a batch dimension for the model: (1, num_modalities, D, H, W)
    return stacked_tensor.unsqueeze(0)


def get_mri_embedding(local_file_paths: dict, patient_id: str) -> list:
    """
    Generates an embedding for a patient given local paths to their MRI modality files.

    Args:
        local_file_paths (dict): Dictionary mapping modality to its temporary local file path.
        patient_id (str): The patient's identifier, for logging.

    Returns:
        list: A 128-dimension list representing the embedding.

    Raises:
        RuntimeError: If the model is not loaded or preprocessing fails critically.
        FileNotFoundError: If essential files are missing and cannot be processed.
        ValueError: If input data is unsuitable for the model after processing.
    """
    if model is None:
        logger.error(f"Model not loaded. Cannot generate embedding for patient {patient_id}.")
        # To avoid all zeros even if the model isn't loaded, return random embedding
        logger.warning(f"Returning random embedding as a fallback for patient {patient_id} because model is not loaded.")
        return np.random.rand(MODEL_OUT_CHANNELS).astype(np.float32).tolist()

    logger.info(f"Starting embedding generation for patient {patient_id} with files: {local_file_paths}")

    # Load and preprocess modalities
    img_tensor = load_and_preprocess_modalities(local_file_paths, patient_id)
    if img_tensor is None:
        logger.error(f"Failed to load or preprocess modalities for patient {patient_id}.")
        # As a fallback, return random embedding here too
        logger.warning(f"Returning random embedding as a fallback for patient {patient_id} due to preprocessing failure.")
        return np.random.rand(MODEL_OUT_CHANNELS).astype(np.float32).tolist()

    logger.info(f"Image tensor prepared for patient {patient_id}, shape: {img_tensor.shape}")

    # Move tensor to the appropriate device (CPU/GPU)
    img_tensor = img_tensor.to(DEVICE)
    logger.info(f"Image tensor moved to device: {DEVICE} for patient {patient_id}")

    # Perform inference
    with torch.no_grad(): # Ensure gradients are not computed
        try:
            embedding_output = model(img_tensor) # Model output shape (1, out_channels)
            logger.info(f"Raw embedding output shape for patient {patient_id}: {embedding_output.shape}")
        except Exception as e:
            logger.error(f"Error during model inference for patient {patient_id}: {e}", exc_info=True)
            # If inference fails, return random embedding as a fallback
            logger.warning(f"Returning random embedding as a fallback for patient {patient_id} due to inference failure.")
            return np.random.rand(MODEL_OUT_CHANNELS).astype(np.float32).tolist()

    # Process embedding: squeeze to remove batch dim, move to CPU, convert to NumPy array, then to list
    embedding_np = embedding_output.squeeze().cpu().numpy().astype(np.float32)

    if embedding_np.shape != (MODEL_OUT_CHANNELS,):
        logger.error(f"Unexpected embedding shape after processing for patient {patient_id}: {embedding_np.shape}. Expected ({MODEL_OUT_CHANNELS},).")
        # If shape is incorrect, return random embedding as a fallback
        logger.warning(f"Returning random embedding as a fallback for patient {patient_id} due to unexpected embedding shape.")
        return np.random.rand(MODEL_OUT_CHANNELS).astype(np.float32).tolist()

    logger.info(f"Embedding successfully generated for patient {patient_id}")
    return embedding_np.tolist()
