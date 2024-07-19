# Image Segmentation
This project aims to perform image segmentation (semantic segmentation) on real life Indian Driving Dataset. The main goal is to fine-tune a pre-trained segmentation model([fcn_resnet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet50.html)) and evaluate its performance on various metrics. The project involves data preprocessing, model training, visualization of loss curves and mean IoU using tensorboard, and reporting the class wise performance of the test set in terms of pixel-wise accuracy, F1-Score and IoU (Intersection Over Union).

## Dataset
The Indian Driving Dataset (IDD) contains images and their corresponding segmentation masks. The dataset is divided into three splits: train, validation, and test.

## Code Flow

### Download the Dataset
- Visit [IDD](https://idd.insaan.iiit.ac.in/) Website.
- Refer to the [IDD](https://insaan.iiit.ac.in/media/publications/idd-650.pdf) Dataset Paper to understand the label structure.

### Unzip the dataset
```
!tar -xzvf /content/drive/MyDrive/"IDD Dataset"/IDDSPLIT.tar.gz -C /content
```

### Convert masks from RGB Format to single-channel format
```
COLOR_DICT = {
    (128, 64, 128): 0,  # Road
    (244, 35, 232): 2,  # Sidewalk
    (220, 20, 60): 4,   # Person
    (255, 0, 0): 5,     # Rider
    (0, 0, 230): 6,     # Motorcycle
    (119, 11, 32): 7,   # Bicycle
    (0, 0, 142): 9,     # Car
    (0, 0, 70): 10,     # Truck
    (0, 60, 100): 11,   # Bus
    (0, 80, 100): 12,   # Train
    (102, 102, 156): 14 # Wall
}
```

### Custom Dataset Class
A custom dataset class is defined to handle the loading and preprocessing of the images and their corresponding segmentation masks.

### Data Loaders
* Create data loaders for train, validation, and test splits using PyTorch.
* Resize the images to `512 x 512` during preprocessing (original resolution is `1920 x 1080`).

### Load Model
* Load and train the `fcn_resnet50` model using pre-trained network weights.
* Defining loss function and optimizer

### TensorBoard Visualization
Visualizing the loss curves and mean IoU on TensorBoard for better insights into the training process.

## Model Evaluation
* Evaluating the model's performance on the test set in terms of pixel-wise accuracy, F1-Score, and IoU (Intersection Over Union).
* Precision, Recall, and Average Precision (AP): Computing these metrics using IoUs within the range [0, 1] with a 0.1 interval size.

## Visualization
* For each class in the test set, visualizing 3 images with IoU ≤ 0.5, showing the predicted and ground truth masks.


