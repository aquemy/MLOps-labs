# MLflow

[MLflow](https://mlflow.org) is a library that supports many elements of the MLOps process, including:

- tracking different versions of experiments (_MLflow Tracking_)
- implementation of the ML model in the form of an artifact for reuse and sharing (_MLflow Projects_)
- management of models built based on various libraries (_MLflow Projects_)
- providing a central repository of models along with managing their life cycle (_MLflow Project Repository_)

One of the interesting features of MLflow is that the library is completely agnostic to libraries for creating ML models. All functionality is available from the REST API and as a set of command line commands, and there are also APIs for Python, Java, and R.

A fundamental concept within `mlflow` is **artifact**. This is any project-related file or directory stored in an external repository. Artifacts can be logged into the repository, and they can also be downloaded and saved to the repository. Artifacts can be objects on a local disk, but they can also be files stored on S3, in HDFS, models with versions, etc.

`mlflow` usage scenarios include:

- for individual researchers and engineers: ability to track training on local machines, maintain multiple configuration versions, convenient storage of models prepared in various architectures
- for _data science_ teams: ability to compare the results of different algorithms, unification of terminology (names of scripts and parameters), sharing of models
- for large organizations: sharing and reusing models and designs, exchanging knowledge, facilitating process productization
- for MLOps: ability to deploy models from various libraries as simple files in the operating system
- for researchers: ability to share and run GitHub repositories

In the below example, we will build a linear regression model to predict wine quality.

### Experiment tracking

Records describing individual runs can be stored:
- in a local directory
- in a database (MySQL, SQLite, PostgreSQL)
- on an HTTP server running MLFlow
- in a Databricks workspace

Go to the folder `wine-quality` and create a `train.py` file and include the following code in it:

```python
import plac # note(aquemy): I would use `typer` right now

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from pathlib import Path

import mlflow
import mlflow.sklearn

@plac.opt('input_file', 'Input file with training data', Path, 'i')
@plac.opt('alpha', 'Alpha parameter for ElasticNet', float, 'a')
@plac.opt('l1_ratio', 'L1 ratio parameter for ElasticNet', float, 'l')
def main(input_file: Path, alpha: float=0.5, l1_ratio: float=0.5):

    assert input_file, "Please provide a file with the training data"

    df = pd.read_csv(input_file, sep=';')

    df_train, df_test = train_test_split(df, train_size=0.8)

    X_train = df_train.drop(['quality'], axis=1)
    X_test = df_test.drop(['quality'], axis=1)
    y_train = df_train['quality']
    y_test = df_test['quality']

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)

        print(f"ElasticNet(alpha={alpha},l1_ratio={l1_ratio}): RMSE={rmse}, MAE={mae}, R2={r2score}")

        mlflow.log_param('alpha', alpha)
        mlflow.log_param('l1_ratio', l1_ratio)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2score', r2score)


if __name__ == "__main__":
    plac.call(main)
```

View the contents of the data file

```bash
cat winequality.csv
```

Check the correct functioning of the script by running it from the command line

```bash
python train.py --help

python train.py --input-file data/winequality.csv --alpha 0.4 --l1-ratio 0.75
```

Observe the structure of the directory `mlruns` 

```bash
tree mlruns
```

Run the training script several times, using different parameter values.

Start the MLflow server and view the information collected about the experiment at [localhost:5000](http://localhost:5000)

```bash
mlflow ui -p 5000 -h 0.0.0.0
```

Return to the command line and run a series of experiments, testing different combinations of parameters. Beforehand, make sure your terminal language settings are English (decimal point causes an error).

```bash
export LANG=en_US

for a in $(seq 0.1 0.1 1.0)
    do
        for l in $(seq 0.1 0.1 1.0)
        do
            python train.py -i data/winequality.csv -a $a -l $l
        done
    done
```

Restart the `mlflow` server and explore the results. Select all runs and add to comparison. Check the available visualizations of individual measures.

Modify the `train.py` file by adding, after the metrics logging, the model logging. To do this, add the following line:

```python
mlflow.sklearn.log_model(lr, 'model')
```

Then run the training once and see the results. See in what form the model has been saved in the repository.

Add a fragment in the code that assigns a tag to a given experiment and observe the tags in the repository.

```python

# set a single tag
mlflow.set_tag('version','0.9')

# set a list of tags
mlflow.set_tags({
    'author': 'Alexandre Quemy',
    'date': '01.09.2023',
    'release': 'candidate'
})
```

Check how incorrect training runs are saved in the repository. To do this, run the script with the incorrect parameter call.

### Automatic tracking of parameters and metrics

`mlflow` can automatically track parameter and metric values for many popular ML libraries. Change the contents of the `train.py` file as follows:

- add import of the `MlflowClient` class from `mlflow.tracking` module
- comment out all code running in the context manager `mlflow.start_run()`
- immediately after dividing into training and testing sets, add the following code:

```python
mlflow.sklearn.autolog()
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

with mlflow.start_run() as run:
    lr.fit(X_train, y_train)

autolog(mlflow.get_run(run_id=run.info.run_id))
```

- add the following function above the code:

```python
def autolog(run):

    tags = {
        k: v 
        for k, v in run.data.tags.items() 
        if not k.startswith("mlflow.")
    }

    artifacts = [
        f.path 
        for f 
        in MlflowClient().list_artifacts(run.info.run_id, "model")
    ]

    print(f"run_id: {run.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {run.data.params}")
    print(f"metrics: {run.data.metrics}")
    print(f"tags: {tags}")
```

Run the training code again and observe the result.

### Grouping runs into experiments

`mlflow` allows you to group multiple runs into named experiments. The first step is to create a named experiment. For instance:

```bash
mlflow experiments create --experiment-name simple-regression
```

Repeat the previous runs, first setting the content of the `MLFLOW EXPERIMENT NAME` environment variable

```bash
export MLFLOW_EXPERIMENT_NAME=simple-regression

python train.py -i data/winequality.csv
python train.py -i data/winequality.csv -a 0.1
python train.py -i data/winequality.csv -l 0.9
```
and view the result in the repository.

Instead of an environment variable, it is possible to use the parameter `--experiment_name` when invoking the `mlflow experiment run` command.

### Packing a model

In the next step, we will build an entire package containing code to train a simple model. Create a `regression` directory and create two files in it: `MLProject` and `conda.yaml`.

The `MLProject` file contains the MLflow project definition. Place the following content in it:

```
name: linear_regression_example

conda_env: conda.yaml

entry_points:
    main:
        command: "python train.py"
```

The `conda.yaml` file contains the definition of the environment in which the code will run.

```yaml
name: regression-example
channels:
  - defaults
  - anaconda
  - conda-forge
dependencies:
  - python=3.7
  - scikit-learn
  - pip
  - pip:
    - mlflow>=1.
```

The `train.py` file contains the model training code. In this case, it is a very simple code that trains a classifier on a toy example.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":

    X = np.arange(-100,100).reshape(-1, 1)
    y = X**2

    lr = LinearRegression()
    lr.fit(X, y)

    score = lr.score(X, y)

    print(f"Score: {score}")

    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")

    print(f"Model saved in run {mlflow.active_run().info.run_uuid}")
```

Run the package by issuing the following command and specifying the conda location:

```bash
export MLFLOW_CONDA_HOME=/root/miniconda3

mlflow run regression
``` 

### Running the model directly from the repository

In the next step, we will save this package as a Git repository and run the experiment directly from the repository

- create a remote repository on GitHub (e.g. called `mlflow_example`
- enter the `regression` directory and initialize the repository with the `git init` command
- add the contents of the directory to the repository with the `git add .` command
- create the first commit with the command `git commit -m "feat: MLflow experiment repo created"`
- copy the remote repository URL
- add the remote repository to your local repository configuration

```bash
git remote add origin <url remote repository>
git remote -v
```
- push local changes to the remote repository with `git push origin main`
- run the experiment in the remote repository with the command `mlflow run <remote repository url>`
  
### Packing the model with parameters

In the next example, we will create an experiment that requires parameters. Create a `wine-quality` directory and create two files in it: `MLProject` and `conda.yaml`.

In the file `MLProject`, place the following content:


```
name: wine_quality_model
conda_env: conda.yaml

entry_points:
    main:
        parameters:
            input_file: {type: str}
            alpha: {type: float, default=0.5}
            l1_ratio: {type: float, default=0.5}
        command: "python train.py -i {input_file} -a {alpha} -l {l1_ratio}"
```

In the file `conda.yaml`, put the following content:

```yaml
name: wine_quality_model
channels:
    - defaults
dependencies:
    - python=3.7
    - scikit-learn
    - pip
    - pip:
        - mlflow>=1.23
        - plac
```

Also copy the file `train.py` and the data file to the `wine-quality` directory.

Set the `MLFLOW_CONDA_HOME` variable to point to your Conda installation. Use `mlflow` to run the package:

```bash
mlflow run wine-quality -P input_file=data.csv -P alpha=0.12 -P l1_ratio=0.79
```

### Serving the model

The model packaged by MLFlow can be easily served. In the repository, have a look for the metadata of one the run. Notice the presence of two files: a bundled model and a text file with metadata. Read the metadata and note the run ID (`run_id`):

Start serving the model by installing the `pyenv` package and issuing the command

```bash
curl https://pyenv.run | bash
export PATH=$HOME/.pyenv/bin:$PATH

mlflow models serve -m "runs:/<run_id>/model" -p 5000 -h 0.0.0.0
```

For instance:
```bash
mlflow models serve -m "runs:/02c9f02d80314a0a805a92f81e24153e/models -p 5000 -h 0.0.0.0
```

Using the REST API, make a prediction by issuing a command

```bash
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://localhost:5000/invocations
```

### Task

Download the dataset from [World Happiness Report 2021](https://www.kaggle.com/ajaypalsinghlo/world-happiness-report-2021).

Using `mlflow`, prepare a package containing the code for training a model. Build a [decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) model and try to conduct an experiment that will allow you to choose the best parameter values:

- maximum tree depth
- a measure of the division point assessment
- minimum number of instances in a leaf

In the experiment, use the metrics of mean absolute error, mean squared error, and the coefficient of determination R2.
