import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import os

class MlflowHelper():
    """
    低レベルAPI の MlflowClient の wrapper class
    """
    def __init__(self, tracking_uri, experiment_name):
        """
        Args:
            tracking_uri : mlflow の結果を保存するディレクトリまでの path
            experiment_name : 実験名
        """

        mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()

        # create experiment
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except Exception as e:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.experiment = self.client.get_experiment(self.experiment_id)
        self.tracking_uri = tracking_uri

        # metrics
        self.running_metrics_dict={}
        self.metrics_dict={}
    
    def create_run(self, tags=None):
        self.run = self.client.create_run(self.experiment_id, tags=tags)
        self.run_id = self.run.info.run_id
        self._make_output_dirs()
    
    def _make_output_dirs(self):
        output_root_path = f'{self.tracking_uri}/{self.experiment_id}/{self.run_id}'
        for dir_name in ["save",]:
            os.makedirs(f'{output_root_path}/{dir_name}/', exist_ok=True)
    
    def set_terminated(self):
        self.client.set_terminated(self.run_id)
    
    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)
    
    def log_params(self, dict):
        for key, value in dict.items():
            self.log_param(key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_metric_epoch(self, key, value, epoch):
        self.client.log_metric(self.run_id, key, value, step=epoch)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def log_dict(self, dictionary, file):
        self.client.log_dict(self.run_id, dictionary, file)
    
    def log_figure(self, figure, file):
        self.client.log_figure(self.run_id, figure, file)

    def log_artifact(self, artifact) :
        """
        local file を artifacts ディレクトリに copy して保存する (not symbolic link)
        """
        self.client.log_artifact(self.run_id, artifact)


    # ----------
    #
    def running_update(self, key, value):
        if self.running_metrics_dict.get(key, None) is None:
            self.running_metrics_dict[key] = [value,]
        else :
            self.running_metrics_dict[key].append(value)

    def update(self, key, value):
        self.metrics_dict[key] = value

    def batch_update(self, key):
        self.metrics_dict[key] = np.mean(self.running_metrics_dict[key])
        self.running_metrics_dict[key] = []

    def batch_update_all(self):
        for key in self.running_metrics_dict.keys():
            self.metrics_dict[key] = np.mean(self.running_metrics_dict[key])
            self.running_metrics_dict[key] = []

    def print_metric(self, keys):
        text = ""
        for key in keys:
            value = self.metrics_dict[key]
            text += f"{key} : {value}, "
        print(text)

    def save_metric(self, idx_fold, epoch):
        for key, value in self.metrics_dict.items():
            self.log_metric_epoch(f"{key}_f{idx_fold}", value, epoch=epoch)
        self.metrics_dict = {} 