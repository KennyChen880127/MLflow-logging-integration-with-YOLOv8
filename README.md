<div align="center">
<h1>
<b>
MLflow-logging-integration-with-YOLOv8
</b>
</h1>
</div>

This repo integrates the MLOps Tool MLflow into the YOLOv8 training process, allowing easy and fast retrieval of each training parameter using its Python API for subsequent applications.
 
# MLflow
[MLflow](https://mlflow.org/)  is an end-to-end platform for model management that can achieve the following:

1. Track experiment data and parameters to compare which set of training parameters yields better results.
2. Package the code for training models in a presentable format, facilitating environment sharing with other data scientists.
3. Manage file records of various types of models for subsequent model deployment operations.
4. Register models and perform model deployment operations using CLI commands.

## Steps to run Code

* Create and activate a virtual environment using Anaconda:

        conda create --name yolov8 python=3.8 -y
        conda activate yolov8

* Install the required packages:

        pip install pyinstaller ultralytics mlflow

* Clone the repository

        git clone https://github.com/KennyChen880127/MLflow-logging-integration-with-YOLOv8.git

* Configure MLflow-related parameters:

```Python
os.environ["MLFLOW_TRACKING_URI"] = "Configure the mlruns path for storing data."
experiment_name = "Set the experiment name."
run_name = "The run_name is the name of the execution task."
```

* Then set the parameters for the YOLOv8 training mode. This project includes an example dataset of safety vest data downloaded from [Roboflow](https://drive.google.com/file/d/14ZI__51kQnuKs2eZLQV9Nn9ZXOuQvAL3/view?usp=drive_link) for testing purposes. You can use this training set to observe the results:

        python train.py

* When starting training, you can open the Anaconda Prompt terminal, activate the yolov8 environment, and navigate to the project directory:

        conda activate yolov8
        cd MLflow-logging-integration-with-YOLOv8


* Start MLflow and you can see the configuration of your parameters in the Overview section and check the current training status in the Model metrics section:

        mlflow ui


* After training is completed, you can obtain all the trained parameters using get_mlflow_para.py:

        python get_mlflow_para.py

* The example content roughly is:
```python
{'metrics': {'lr/pg0': 0.0006346000000000005,
             'lr/pg1': 0.0006346000000000005,
             'lr/pg2': 0.0006346000000000005,
             'metrics/mAP50-95B': 0.6819091425676801,
             'metrics/mAP50B': 0.9359368246681663,
             'metrics/precisionB': 0.9301563045008876,
             'metrics/recallB': 0.889915344460799,
             'train/box_loss': 0.45572,
             'train/cls_loss': 0.27814,
             'train/dfl_loss': 0.89044,
             'val/box_loss': 1.05183,
             'val/cls_loss': 0.52325,
             'val/dfl_loss': 1.23342},
 'params': {'agnostic_nms': 'False',
            'amp': 'True',
            'augment': 'False',
            'auto_augment': 'randaugment',
            'batch': '80',
            'box': '7.5',
            'cache': 'False',
            'cfg': 'None',
            'classes': 'None',
            'close_mosaic': '10',
            'cls': '0.5',
            'conf': 'None',
            'copy_paste': '0.0',
            'cos_lr': 'False',
            'crop_fraction': '1.0',
            'data': 'vest_dataset/vest.yaml',
            'degrees': '0.0',
            'deterministic': 'True',
            'device': 'None',
            'dfl': '1.5',
            'dnn': 'False',
            'dropout': '0.0',
            'dynamic': 'False',
            'embed': 'None',
            'epochs': '500',
            'erasing': '0.4',
            'exist_ok': 'False',
            'fliplr': '0.5',
            'flipud': '0.0',
            'format': 'torchscript',
            'fraction': '1.0',
            'freeze': 'None',
            'half': 'False',
            'hsv_h': '0.015',
            'hsv_s': '0.7',
            'hsv_v': '0.4',
            'imgsz': '640',
            'int8': 'False',
            'iou': '0.7',
            'keras': 'False',
            'kobj': '1.0',
            'label_smoothing': '0.0',
            'line_width': 'None',
            'lr0': '0.01',
            'lrf': '0.01',
            'mask_ratio': '4',
            'max_det': '300',
            'mixup': '0.0',
            'mode': 'train',
            'model': 'yolov8n.pt',
            'momentum': '0.937',
            'mosaic': '1.0',
            'multi_scale': 'False',
            'name': 'train10',
            'nbs': '64',
            'nms': 'False',
            'opset': 'None',
            'optimize': 'False',
            'optimizer': 'auto',
            'overlap_mask': 'True',
            'patience': '100',
            'perspective': '0.0',
            'plots': 'True',
            'pose': '12.0',
            'pretrained': 'True',
            'profile': 'False',
            'project': 'None',
            'rect': 'False',
            'resume': 'False',
            'retina_masks': 'False',
            'save': 'True',
            'save_conf': 'False',
            'save_crop': 'False',
            'save_dir': 'runs\\detect\\train10',
            'save_frames': 'False',
            'save_hybrid': 'False',
            'save_json': 'False',
            'save_period': '-1',
            'save_txt': 'False',
            'scale': '0.5',
            'seed': '0',
            'shear': '0.0',
            'show': 'False',
            'show_boxes': 'True',
            'show_conf': 'True',
            'show_labels': 'True',
            'simplify': 'False',
            'single_cls': 'False',
            'source': 'None',
            'split': 'val',
            'stream_buffer': 'False',
            'task': 'detect',
            'time': 'None',
            'tracker': 'botsort.yaml',
            'translate': '0.1',
            'val': 'True',
            'verbose': 'True',
            'vid_stride': '1',
            'visualize': 'False',
            'warmup_bias_lr': '0.0',
            'warmup_epochs': '3.0',
            'warmup_momentum': '0.8',
            'weight_decay': '0.0005',
            'workers': '0',
            'workspace': '4'},
 'tags': {'mlflow.runName': 'train1',
          'mlflow.source.name': 'd:/Project/MLflow-logging-integration-with-YOLOv8/train.py',
          'mlflow.source.type': 'LOCAL',
          'mlflow.user': 'kenny'}}
```
