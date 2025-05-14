PCB Defects Detector using YOLOX in MATLAB

This project implements an automated system for detecting and classifying defects in printed circuit boards (PCBs) using the YOLOX object detection model within MATLAB. The system identifies six common PCB defects:

Missing Hole

Mouse Bite

Open Circuit

Short

Spur

Spurious Copper

📁 Project Structure

main.m: Main script to run the detection pipeline.

readPCBDefectAnnotations.m: Function to parse XML annotations into MATLAB structures.

xml2struct.m: Utility to convert XML files to MATLAB structs.

trainedPCBDetector.mat: Pretrained YOLOX model for PCB defect detection.

PCB-DATASET-master/: Directory containing the dataset images and annotations.

📊 Dataset

The dataset comprises 1,386 images of PCB elements with synthesized defects. Each image contains multiple defects of the same category in different locations. Annotations are provided in XML format, detailing bounding boxes for each defect. The dataset includes six types of defects: missing hole, mouse bite, open circuit, short, spur, and spurious copper.
MathWorks

🚀 Getting Started

Prerequisites
MATLAB R2023b or later

Computer Vision Toolbox

Deep Learning Toolbox

Image Processing Toolbox

Installation
Clone the repository:
git clone https://github.com/AbdelazizChamsine1/PCB-defects-detector.git

2. Open MATLAB and navigate to the cloned repository directory.
3. Ensure all required toolboxes are installed.

Running the Detection
1. Load the pretrained detector:
load('trainedPCBDetector.mat');

2. Run the main script:
main.m

This script will process sample images and display detected defects with bounding boxes and labels.

🧠 Model Details

Architecture: YOLOX (You Only Look Once X)

Base Network: CSP-DarkNet-53

Input Size: 800x800x3

Classes: 6 (as listed above)

📈 Performance

The YOLOX model was trained and evaluated on the PCB defect dataset, achieving high accuracy in detecting and classifying defects across all six categories. The model demonstrates real-time performance suitable for automated quality control in manufacturing environments.

🤝 Acknowledgments

MathWorks for providing the Computer Vision and Deep Learning Toolboxes.

The creators of the PCB defect dataset used in this project.
