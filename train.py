from ultralytics import YOLO, settings
import mlflow
import os 

if __name__ == '__main__':
    os.environ["MLFLOW_TRACKING_URI"] = "file:///D:/Project/MLflow-logging-integration-with-YOLOv8/mlruns" # Configure the mlruns path for storing data.
    run_name = "train1"
    experiment_name = "vest-experiment"
    
    print("MLFLOW_TRACKING_URI: ", os.environ.get("MLFLOW_TRACKING_URI"))


    model = YOLO(r'yolov8n.pt') # Load YOLOv8 weights.
    
    # Make sure MLflow is enabled in settings
    settings.update({'mlflow': True})
    settings.reset()
    mlflow.pytorch.autolog(log_models=True)
    mlflow.set_experiment(experiment_name) # Set the experiment name.

    with mlflow.start_run(run_name=run_name) as run: # The run_name is the name of the execution task.

        run_id = run.info.run_id # The run_id is the directory name that will be stored within the MLFLOW_TRACKING_URI path.
        print(run_id)

        # Set YOLOv8 training model parameters.
        model.train(
            data='vest_dataset/vest.yaml',
            epochs=500,
            batch=80,
            imgsz=640,
            workers=0
        )

    mlflow.end_run()
