"""Project hooks."""
import sys
import time
import collections
import os
import pandas as pd
import cloudpickle
import sklearn
from typing import Any, Dict, Iterable, Optional
from dotenv import load_dotenv

import cookiecutter as cookiecutter
from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.versioning import Journal

from dotenv import load_dotenv

load_dotenv()

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

import mlflow
from mlflow.entities.run_status import RunStatus
from mlflow.models.signature import infer_signature

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.versioning import Journal
from kedro.pipeline.node import Node

from {{ cookiecutter.python_package }}.extras.config import CustomTemplatedConfigLoader

PYTHON_VERSION = "{major}.{minor}.{micro}".format(
    major=sys.version_info.major,
    minor=sys.version_info.minor,
    micro=sys.version_info.micro,
)

class ProjectHooks:
    @hook_impl
    def register_config_loader(
            self,
            conf_paths: Iterable[str],
            env: str,
            extra_params: Dict[str, Any],
    ) -> CustomTemplatedConfigLoader:
        return CustomTemplatedConfigLoader(
            conf_paths, globals_pattern="*globals.yml", globals_dict=extra_params
        )

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )


class MLFlowTrackingHooks:
    """Namespace for grouping all model-tracking hooks with MLflow together."""

    @hook_impl
    def after_catalog_created(
        self,
        conf_catalog: Dict[str, Any],
        feed_dict: Dict[str, Any],
        load_versions: Dict[str, str],
    ) -> None:
        parameters = feed_dict.get("parameters", {})
        mlflow_params = parameters.get("mlflow", {})

        self._client = mlflow.tracking.MlflowClient()
        self._exp_name = mlflow_params["experiment_name"]
        self._artifact_location = mlflow_params["artifact_location"]
        self._conf_catalog = conf_catalog
        self._load_versions = load_versions
        self._top_n = 20
        self._conda_env = {
            "name": "env",
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                "python={}".format(PYTHON_VERSION),
                "pip",
            ],
            "pip": [
                "mlflow=={}".format(mlflow.__version__),
                "pandas=={}".format(pd.__version__),
                "scikit-learn=={}".format(sklearn.__version__),
                "cloudpickle=={}".format(cloudpickle.__version__),
            ],
        }

    @hook_impl
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
    ) -> None:
        try:
            self._exp_id = mlflow.get_experiment_by_name(self._exp_name).experiment_id
        except:
            self._exp_id = mlflow.create_experiment(
                name=self._exp_name, artifact_location=self._artifact_location
            )
        finally:
            mlflow.set_experiment(experiment_name=self._exp_name)
        self._run = mlflow.start_run(experiment_id=self._exp_id)
        mlflow.set_tags(run_params)
        self._target = run_params["pipeline_name"] or self._default_target

    @hook_impl
    def after_node_run(
        self, node: Node, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> None:
        parameters = {
            key.split(":", 1)[-1]: value
            for key, value in inputs.items()
            if key.startswith("params")
        }
        mlflow.log_params(self._flatten(parameters, sep="."))

        if node._func_name == "h2o_models_test":
            target = self._target.upper()

            df_test_name = "MIN_{}_TS_STD".format(target)
            mdl_name = "MDL_{}_ELECTED_MODEL".format(target)
            result_name = "MOU_{}_ELECTED_RESULTS".format(target)

            df_test = inputs[df_test_name]
            model = outputs[mdl_name]
            results = outputs[result_name]

            # Plots (Explanation)
            explanation = h2o.explain(
                models=model,
                frame=h2o.H2OFrame(df_test),
                top_n_features=self._top_n,
                include_explanations=[
                    "varimp",
                    "residual_analysis",
                    "shap_summary",
                    "pdp",
                ],
                render=False,
            )
            self._log_explanation(explanation)

            # Metrics
            mlflow.log_metric("MAE", results.at[0, "mae"])

    @hook_impl
    def after_pipeline_run(self, catalog: DataCatalog) -> None:
        target = self._target.upper()
        df_name = "MIN_{}".format(target)
        res_name = "MOU_{}_ELECTED_RESULTS".format(target)
        pred_name = "MOU_{}_ELECTED_PREDICTION".format(target)
        enc_name = "MIN_{}_ENCODINGS".format(target)
        scl_name = "MIN_{}_SCALER".format(target)
        mdl_name = "MDL_{}_ELECTED_MODEL".format(target)

        df = catalog.load(name=df_name, version=self._load_versions.get(df_name))
        results = catalog.load(name=res_name, version=self._load_versions.get(res_name))
        prediction = catalog.load(
            name=pred_name, version=self._load_versions.get(pred_name)
        )

        # DS metrics
        rows, cols = df.shape
        target_col_name = df.columns[-1]

        mlflow.log_param("dataset.rows", rows)
        mlflow.log_param("dataset.cols", cols)
        mlflow.log_param("target_column.name", target_col_name)
        mlflow.log_param("target_column.type", df.dtypes[target_col_name])
        mlflow.log_param("target_column.uniques", len(df[target_col_name].unique()))

        # Artifacts
        artifacts = {
            "Encoding": self._get_dataset_path_from_config(enc_name),
            "Scaler": self._get_dataset_path_from_config(scl_name),
            "Model": os.path.join(
                self._get_dataset_path_from_config(mdl_name),
                f"{results.at[0, 'model_id']}.zip",
            ),
            "Genmodel": os.path.join(
                self._get_dataset_path_from_config(mdl_name),
                "h2o-genmodel.jar",
            ),
        }

        # Model
        flavor.log_model(
            artifact_path="model",
            code_path=[os.path.dirname(flavor.__file__)],
            conda_env=self._conda_env,
            python_model=flavor.infos_h2o._H2OModel(),
            artifacts=artifacts,
            registered_model_name=self._target,
            signature=infer_signature(df.iloc[:, :-1], prediction),
        )

        mv = self._client.get_latest_versions(name=self._target, stages=["None"])[0]
        self._client.transition_model_version_stage(
            name=self._target,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=True,
        )

        mlflow.set_tag("status", "Pipeline executed successfully.")
        mlflow.end_run()

    @hook_impl
    def on_node_error(self, error: Exception) -> None:
        mlflow.set_tag(key="status", value="Pipeline terminated with error.")
        mlflow.set_tag(
            key="error",
            value=f"{error.__class__.__module__}.{error.__class__.__name__}: {error}",
        )
        mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))

    @hook_impl
    def on_pipeline_error(self, error: Exception) -> None:
        mlflow.set_tag("status", "Pipeline terminated with error.")
        mlflow.set_tag(
            "Error", f"{error.__class__.__module__}.{error.__class__.__name__}: {error}"
        )
        mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))

    def _flatten(
        self, d: collections.MutableMapping, parent_key: str = "", sep: str = "_"
    ) -> Dict[str, Any]:

        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self._flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _log_explanation(self, explanation: dict) -> None:
        fpath = "plots"
        img_format = "png"
        for name, info in explanation.items():
            _fpath = "/".join([fpath, name])

            description = str(info.get("description", ""))
            mlflow.log_text(
                text=description, artifact_file="/".join([_fpath, "description.txt"])
            )

            plots = info.get("plots", dict())
            if isinstance(plots, H2OExplanation):
                _plots = self._flatten(plots, sep=".")
                for subname, figure in _plots.items():
                    __fpath = "/".join([_fpath, *subname.split(".")])
                    __fpath = ".".join([__fpath, img_format])
                    mlflow.log_figure(figure=figure, artifact_file=__fpath)
            else:
                _fpath = "/".join([_fpath, name])
                _fpath = ".".join([_fpath, img_format])
                mlflow.log_figure(figure=plots, artifact_file=_fpath)

    def _get_dataset_path_from_config(self, dataset_name: str) -> str:
        conf = self._conf_catalog[dataset_name]
        fpath = conf["filepath"]
        if conf.get("versioned", False):
            fname = fpath.split("/")[-1]
            fpath = os.path.join(
                fpath, self._load_versions.get(dataset_name, ""), fname
            )
        return fpath


class PipelineMonitoringHooks:
    def __init__(self):
        self.registry = CollectorRegistry()
        self._metrics = {
            # Node Execution Time
            "NET": Gauge(
                name="node_exec_time",
                documentation="Execution Time of a node",
                labelnames=["target", "node_name"],
                unit="seconds",
                registry=self.registry,
            ),
            # DataSet Size
            "DSS": Gauge(
                name="dataset_size",
                documentation="Size of a dataset",
                labelnames=["target", "ds_name"],
                unit="bytes",
                registry=self.registry,
            ),
            # DataSet Shape
            "DSROWS": Gauge(
                name="dataset_nrows",
                documentation="Number of rows of a dataset",
                labelnames=["target", "ds_name"],
                registry=self.registry,
            ),
            "DSCOLS": Gauge(
                name="dataset_ncols",
                documentation="Number of columns of a dataset",
                labelnames=["target", "ds_name"],
                registry=self.registry,
            ),
        }
        self._timers = {}
        self._datasets = set()

    @hook_impl
    def before_node_run(self, node: Node) -> None:
        self._timers[node.name] = time.time()

    @hook_impl
    def after_node_run(
        self, node: Node, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> None:
        target = node.namespace

        self._timers[node.name] -= time.time()
        _node_name = node.name.lower().replace(" ", "_").replace("-", "_")
        self._metrics["NET"].labels(target=target, node_name=_node_name).set(
            abs(self._timers[node.name])
        )

        for dataset_name, data in {**inputs, **outputs}.items():
            if dataset_name.startswith("param"):
                continue
            if dataset_name in self._datasets:
                continue
            self.set_dataset_size(str(target), dataset_name, data)
            self._datasets.add(dataset_name)

        push_to_gateway("http://127.0.0.1:9091", job="kedro", registry=self.registry)

    def set_dataset_size(self, target: str, dataset_name: str, data: Any) -> None:
        size = sys.getsizeof(data)
        _ds_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
        self._metrics["DSS"].labels(target=target, ds_name=_ds_name).set(size)