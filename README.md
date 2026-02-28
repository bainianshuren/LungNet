LungNet: A lightweight pulmonary nodule detection algorithm Based on multi-dimensional dynamic convolution and anatomy-aware network

Project Description
This project implements a lightweight pulmonary nodule detection model LungNet based on the improved YOLOv11. The model optimizes the detection performance of small nodules and under complex backgrounds while maintaining light weight by designing a Multi-dimensional Dynamic Convolution Module (MDCN), a Dynamic Weight-Assigned Atrous Convolution (DWA-Conv), and an Anatomy-Aware Dual-Path Fusion Network (AAF-Net).

The project includes complete code for model training, testing, ablation experiments, and public dataset preprocessing, which can directly reproduce all experimental results in the paper. The core innovations of LungNet are reflected in adaptive multi-scale feature extraction, efficient feature fusion, and anatomical context modeling, which achieve a good balance between detection accuracy and model efficiency, and have potential clinical application value.

Dataset Information
Supported Datasets
1. LUNA16 Dataset
   - Source: Subset of the Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI)
   - Official Download: https://luna16.grand-challenge.org/
   - Preprocessing Specification: 3D CT volume data is converted to 2D slices, cropped to 330×330 resolution, and pixel values are normalized to [0,1]; a total of 1186 images are obtained, divided into training and test sets at a ratio of 7:3.

2. Lung-PET-CT-Dx Dataset
   - Source: The Cancer Imaging Archive (TCIA), collected by the Second Affiliated Hospital of Harbin Medical University
   - Official Download: https://doi.org/10.7937/TCIA.ABDE-EF92
   - Preprocessing Specification: CT images (512×512 resolution) and PET images (200×200 resolution) are format-converted and label-matched; divided into training and test sets at a ratio of 7:3.

Data Annotation Format
The nodule annotation adopts the classic object detection bounding box format `(xmin, ymin, xmax, ymax, class)`. The project provides a label file conversion script to adapt to the input requirements of the LungNet model.

Code Information
Core Framework
Python + PyTorch (modular design, low coupling between modules, easy to modify and expand)

Code Structure
LungNet/
├── models/        Core code of LungNet, including MDCN, DWA-Conv, AAF-Net and overall network construction
├── data/          Dataset preprocessing, loading and augmentation scripts
├── train/         Model training script (learning rate scheduling, optimizer configuration, loss function definition)
├── test/          Model testing and metric calculation script (mAP, mRecall, FPS, etc.)
├── ablation/      Ablation experiment code, verify the effectiveness of each module independently
├── utils/         Tool functions (logging, weight saving, visualization, metric calculation)
├── weights/       Pre-trained optimal weight file of LungNet (can be directly used for testing and inference)
├── README.md      Project description and usage instructions
└── requirements.txt Dependent library list


Compatibility
- Support Windows/Linux operating system
- Compatible with Python 3.8 and above

Usage Methods
 1. Environment Preparation
Install all dependent libraries with one click through the `requirements.txt` file in the root directory:
bash
pip install -r requirements.txt


2. Dataset Preparation
1. Download the original LUNA16 and Lung-PET-CT-Dx datasets from the official links above;
2. Run the preprocessing script to complete format conversion, cropping and normalization:
bash
Preprocess LUNA16 dataset
python data/preprocess.py --dataset LUNA16 --raw_path [Your raw LUNA16 path] --save_path [Your save path]

Preprocess Lung-PET-CT-Dx dataset
python data/preprocess.py --dataset Lung-PET-CT-Dx --raw_path [Your raw PET-CT path] --save_path [Your save path]


3. Model Training
Run the training script, support custom dataset, batch size, training epochs and other parameters:
bash
python train/train.py --dataset LUNA16 --data_path [Your preprocessed data path] --batch_size 8 --epochs 50 --lr 1e-4

- Key training parameters: initial learning rate 1e-4 (adjusted to 1e-5 after 15 epochs), Adam optimizer, CIoU loss function, batch size 8, total training epochs 50.

4. Model Testing
Load the pre-trained optimal weight file, run the test script, and automatically calculate and output evaluation metrics:
bash
python test/test.py --dataset LUNA16 --data_path [Your preprocessed data path] --weight_path weights/best_LungNet.pth

- The script outputs mAP@0.5, mAP@0.5:0.95, mRecall, FPS, Params, GFLOPs and other metrics by default, consistent with the experimental settings in the paper.

5. Ablation Experiment
Verify the effectiveness of a single module (MDCN/DWA-Conv/AAF-Net) or combined modules independently:
bash
Verify the combined effect of MDCN and DWA-Conv modules
python ablation/ablation.py --dataset LUNA16 --data_path [Your preprocessed data path] --modules MDCN DWA-Conv

- The ablation experiment is based on the YOLOv11 baseline model, and 7 groups of experimental configurations are supported (3 single modules, 3 double-module combinations, 1 three-module combination).

6. Model Inference
Support inference and visualization of single/batch CT images, and save the detection results (bounding box + confidence):
bash
python test/infer.py --img_path [Your single/batch image path] --weight_path weights/best_LungNet.pth --save_path [Your inference result save path]


Dependencies (Python Libraries)
| Library Name      | Version   | Function Description                                                                 |
|-------------------|-----------|--------------------------------------------------------------------------------------|
| Python            | 3.8.18    | Basic development environment                                                        |
| PyTorch           | 1.14.0    | Deep learning framework (model building/training)                                    |
| Torchvision       | 0.15.1    | Computer vision toolkit (data augmentation)                                          |
| NumPy             | 1.24.3    | Numerical calculation and array processing                                            |
| Pandas            | 1.5.3     | Data processing and label file parsing                                                |
| OpenCV-Python     | 4.8.0     | Image reading, preprocessing and visualization                                        |
| Matplotlib        | 3.7.1     | Experimental result visualization and curve plotting                                  |
| Scikit-learn      | 1.2.2     | Metric calculation and dataset division                                              |
| tqdm              | 4.65.0    | Training process progress bar                                                        |
| Pillow            | 9.5.0     | Image format processing                                                              |

All dependent libraries and their specified versions are listed in `requirements.txt` to ensure environment consistency and avoid version compatibility issues.

Data Processing / Modeling Steps
Data Processing Steps
1. Raw Data Acquisition: Download the original DICOM format CT/PET-CT images and corresponding nodule annotation files from the official dataset channel;
2. Format Conversion: Convert DICOM format to PNG format for easy model reading; convert the original annotation files to the txt format adapted to the YOLO series model;
3. Image Preprocessing:
   - 3D to 2D: Slice the 3D CT volume data of LUNA16 and extract slices containing pulmonary nodules;
   - Cropping & Normalization: Crop images to the specified resolution and normalize pixel values to [0,1];
   - Data Augmentation: Adopt random flipping, rotation, scaling, brightness adjustment and other strategies in the training stage to improve the generalization ability of the model;
4. Dataset Division: Randomly divide both datasets into training and test sets at a ratio of 7:3 (fixed division result to ensure experiment reproducibility);
5. Data Loading: Build a data loader using PyTorch's DataLoader with a batch size of 8 and support multi-threaded loading.

Modeling & Training Steps
1. Model Construction: Based on the YOLOv11 baseline framework, replace the backbone network with MDCN, replace the neck feature pyramid with DWA-Conv, and add AAF-Net as the pre-module of the detection head to build the overall LungNet network;
2. Hyperparameter Setting: Consistent with the paper, initial learning rate 1e-4, adjusted to 1e-5 after 15 epochs, total 50 epochs, Adam optimizer, CIoU loss function;
3. Model Training: End-to-end training on the training set, with early stopping strategy (stop training and save the optimal weight when the validation set mAP does not improve for 5 consecutive epochs);
4. Model Testing: Load the optimal weight on the test set for inference, calculate mAP, mRecall (IoU=0.5 and IoU=0.5:0.95), and count Params, FPS, GFLOPs;
5. Ablation Experiment: Based on YOLOv11, add MDCN/DWA-Conv/AAF-Net modules separately and in combination to verify the effectiveness and synergy of each module;
6. Result Visualization: Visualize the detection results (draw bounding boxes and confidence), and compare the detection effects of different models under complex clinical scenarios.

Citation
If the code or model of this project is used in related research, please cite the original paper:
Wang Y, Xing P Z, Shao J, Hu X S, Shang G B. LungNet: A lightweight pulmonary nodule detection algorithm based on multi-dimensional dynamic convolution and anatomy-aware network. 

License
This project is released under the MIT License, which permits free use, copying, modification, distribution, and secondary development for both commercial and non-commercial purposes, with the only requirement to retain the original author's copyright notice and license statement.

The full license text is available in the `LICENSE` file in the root directory of the repository, and the official description can be viewed at: https://opensource.org/licenses/MIT

Reproducibility Guarantee
1. Fixed Random Seed: Set the random seeds of Python, NumPy and PyTorch in all codes to ensure the consistency of random operations in the training process;
2. Pre-trained Optimal Weights: The repository provides the pre-trained optimal weight file of LungNet, which can directly run the test script without re-training to reproduce all experimental metrics in the paper;
3. Unified Experimental Environment: Clearly specify the version of all dependent libraries and provide a one-click installation script to avoid result deviations caused by environmental differences;
4. Complete Experimental Code: Include all codes for training, testing and ablation experiments without hidden logic, and all parameter settings are completely consistent with the paper;
5. Standardized Preprocessing: Provide the same dataset preprocessing script as the paper, and the input original public dataset can obtain the preprocessed data consistent with the experiment.
