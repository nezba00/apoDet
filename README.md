# Single-Cell Time-Series Preprocessing and Temporal Channel Encoding Pipeline

This repository contains the core pipeline for transforming raw, multi-frame imaging data into specialized, multi-channel datasets ready for deep learning training. The primary function of this pipeline is to **isolate and generate standardized single-cell crops** with enhanced temporal context.

## Core Methodology: Temporal Channel Encoding

The central feature of this pipeline is its ability to take input images that are single-channel (2D + Time) and generate output windows that are multi-channel (3D). This process is critical for leveraging legacy imaging data:

* **Temporal Stacking:** The pipeline stacks images of the *same individual cell* captured at *different sequential timepoints* along the channel axis.
* **Data Enrichment:** This method encodes crucial **temporal context** (e.g., changes over the past $N$ frames) directly into the model's input channels, creating a significantly richer dataset for single-cell classification.
* **Legacy Data Enablement:** This process allows existing single-channel image archives to be utilized effectively for advanced projects they were not originally designed for.

## Model Compatibility and Training Readiness

This pipeline is specifically engineered to generate training data that is immediately compatible with modern deep learning architectures, particularly those utilizing self-supervised learning (SSL) and transformer-based models (e.g., scDINO). By encoding temporal sequences into the channel dimension, the data is pre-optimized for efficient feature extraction by 2D or 3D convolutional and attention layers, enabling downstream tasks like phenotype classification and trajectory prediction without complex recurrent neural networks.

## Key Features and Generated Artifacts

1.  **Single-Cell Crop Generation:** Systematically isolates and crops individual cell events (e.g., apoptotic, healthy, random) based on tracking metadata, creating the primary dataset for model training.
2.  **Multi-Artifact Output:** The pipeline generates and centralizes all necessary artifacts for ML research, including:
    * **Single-Cell Crops** (Multi-channel TIFFs)
    * **Finalized Tracking Data** (Cleaned and gap-filled trajectory records)
    * **Segmentation Masks**
3.  **Robust Tracking Post-Processing:** Implements logic to analyze and fill gaps within initial object tracking data, ensuring continuous and accurate trajectory records.
4.  **Efficient Data Management (Soft Linking):** Employs symbolic linking (soft links) to reference Quality Control (QC) image data rather than duplicating large TIFF files, significantly improving disk I/O and reducing storage requirements.
