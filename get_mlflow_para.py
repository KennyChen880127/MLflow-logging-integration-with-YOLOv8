import mlflow
import pprint

mlruns_path = "D:/Project/MLflow-logging-integration-with-YOLOv8/mlruns" # The directory path of "mlruns".
trackingDir = "file:///" + mlruns_path

runID = "d83e19a6cab247a887fb094ac2b49107" # runID = 'Enter the run_id'

# Retrieve metrics using get_metric_history().
# metricKey = 'lr/pg0'
# metrics = client.get_metric_history(runID, metricKey)

client = mlflow.tracking.MlflowClient(tracking_uri=trackingDir)

run_data_dict = client.get_run(runID).data.to_dictionary()
pprint.pp(run_data_dict) # Beautify the output using pprint.