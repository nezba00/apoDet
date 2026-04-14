# Transforming Timelapses into Single-Cell Multi-Channel ML Datasets
This pipeline transforms raw mono-channel microscopy timelapses into multi-channel single-cell datasets. By stacking temporal frames into the channel dimension, the pipeline produces images that also carry temporal information and can then be used for any kind of feature extraction. The output of the pipeline is also well suited for the use with ML-/DL-approaches and was validated with DINO trained ViTs.

## Key Features
* **Temporal Channel Encoding:** Converts time-series data into 3D volumes where the Channel-Axis represents time ($t_0, t_1, \dots, t_n$).
* **Cell-Centered Cropping:** Automatic centroid alignment across frames to reduce movement noise.
* **Robust-Tracking:** Built-in SAD-based outlier detection to handle Field-of-View jumps.
* **Optional Expert Annotation Matching:** High-precision (>99%) matching of manual labels to tracks for facilitated supervised learning or downstream validation.

## Pipeline Visual Overview

This diagram illustrates the flow from raw data acquisition through the core preprocessing steps—Segmentation, Tracking, Matching, and Cropping—culminating in the final multi-channel, cell-centered dataset.

<br>
<br>

<div align="center">
    <img 
        src="./assets/pipeline_overview.png" 
        width="90%" 
        alt="Image Description: Pipeline Visual Overview"
    />
</div>

<br>
  
---
## Quick Start
### Input Data Requirements
To run the pipeline, you need the following directories somwhere accessible:
```text
input_data/
├── img_data/
│   ├── exp01_001.tif        # 2D+T TIFF stacks
│   └── exp01_002.tif
├── annotations/
│   ├── exp01_001.csv       # Optional: (x, y, t, filename) expert labels
│   └── exp01_002.csv
└── experiment_info/
    └── experiment_info.csv # (Experiment, magnification, Acquisition_frequency(min), Apo_annotation_frequency(min))
```
The directories can be targeted separately in the config and do not need to be moved into one input_data directory as shown above.

### Installation & Setup
Clone the repository and create the environment using the provided `environment.yml`:
```bash
git clone https://github.com/pertzlab/apoDet.git
cd apoDet
conda env create -f environment.yml
conda activate preprocessing
```
The final step before you can produce your own dataset is to customize `config.yml`.
### Configuration Checklist
**Minimum Check:**
* `RUN_NAME`: Defines the name of the output directory
* `EXTERNAL_PATHS`: Paths to source_images_dir/, experiment_info.csv & manual_annotations/ (optional)
* `TARGET_CHANNEL`: Choose which channel you want to use here
**Custom Runs:**
* `MIN_NUC_SIZE`: Defines threshold used for small object filtering after segmentation
* `MAX_TRACKING_DURATION`: Define how much time the crops span (minutes)
* `FRAME_INTERVAL`: Define time between "frames" in the produced crops (e.g. 20 minutes with an interval of 5 lead to 5 channels: t0, t5, t10, t15, t20) 
* `CROPS_PER_TRACK`: Define how many crops are extracted per detected track (-1 also possible if you want all available positions)
* `WINDOW_SIZE`: Define how big in the spatial dimension the crops should be.

If you are ready to run the pipeline you can do this either locally or submit it to a cluster.
```bash
python pipeline.py        # Local execution
sbatch run_pipeline.sh    # Cluster submission (Slurm)
```
---
## Pipeline Stages & Performance
| Stage | Method/Tool | Performance | Description |
|-------|-------------|-------------|-------------|
| 1. Segmentation | StarDist | 11min/file | Segmentation of cell nuclei |
| 2. Tracking | btrack | 7min/file | Generate trajectories |
| 3. Matching | Majority Vote | 0.1s/annot | Success Rate > 99% |
| 4. Cropping | Custom | 70s/file | Generate centered 32x32x5 crops |

The pipeline overall takes less than 10 minutes per GB of input.

---

## Output Data Structure
The generated output is a multi-channel crop $I'(x, y, c)$ where:
$$\text{Output Channel } c_i \text{ is the image of the cell at time } t_0 + i \cdot \Delta t$$



<br>
<br>

<div align="center">
    <img 
        src="./assets/single_cell_crop_example.png" 
        width="30%" 
        alt="Image Description: Single-Cell Crop Example"
    />
</div>

<br>
---

## Downstream Application Example: scDINO Latent Space Exploration
The output from this preprocessing pipeline served as the foundation for self-supervised training of a model with the **[scDINO framework](https://github.com/JacobHanimann/scDINO)**, which extends **[DINO](https://github.com/facebookresearch/dino)** to more than three channels.

### scDINO Training & Analysis Highlights
* **Unsupervised Feature Extraction:** The temporal channel-encoded crops served as input to the scDINO Vision Transformer (ViT). The model learns representations that capture both morphology and temporal dynamics without any labels.
* **Latent Space Exploration (UMAP):**
    * **Biological Clusters (A-D):** Specific regions for **apoptotic cells (B)**, and detailed clusters for various phases of **mitosis**: metaphase to anaphase (A), telophase to cytokinesis (C), and cells captured during/after cytokinesis (D).
    * **Technical/Artifact Clusters (E-H):** Technical failures, underlining the robustness of the temporal encoding. These clusters included crops with **tracking errors** consistently appearing in specific frames (E and F), cells out of the **focal plane (G, epithelial extrusion)**, and crops displaying a characteristic **grainy texture** likely due to imaging artifacts (H).

<br>
<br>

<div align="center">
    <img 
        src="./assets/example_UMAP.png" 
        width="80%" 
        alt="Image Description: Example Downstream Application"
    />
</div>

<br>

---

## Citations and Acknowledgements

This pipeline relies on several fundamental open-source tools and published methodologies:

* **Segmentation** was performed using [StarDist](https://github.com/stardist/stardist) (Schmidt et al., 2018).
* **Tracking** utilized [btrack](https://github.com/quantumjot/btrack) (Ulicna et al., 2021) which was adapted for robustness against FoV jumps.
* **Downstream Application** [scDINO](https://github.com/JacobHanimann/scDINO) (Pfaendler et al., 2023) is an adaptation of the original self-supervised learning model [DINO](https://github.com/facebookresearch/dino) (Caron et al., 2021).

This project was developed as a Master thesis in the [PertzLab](https://github.com/pertzlab) at the University of Bern, Switzerland. If you like this project, are into automated microscopy or interested in dynamic signalling you might like to have a look at some of our other projects:
* **[ARCOS](https://github.com/pertzlab/ARCOS)** automatically detects collective events like waves of protein activity propagating through a tissue. Is also available as a [plug-in for napari](https://github.com/bgraedel/arcos-gui.git). Also check out the newest member in the ARCOS-ecosystem, **[ARCOS.px](https://github.com/pertzlab/arcosPx-napari)**
* **[rtm-pymmcore](https://github.com/pertzlab/rtm-pymmcore)** lets you communicate with your microscope in real time in python.
* **[fabscope](https://github.com/hinderling/fabscope)** turns your microscope into a 3d printer.
