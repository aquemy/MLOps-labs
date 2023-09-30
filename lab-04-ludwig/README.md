# Ludwig

[Ludwig](https://ludwig-ai.github.io/ludwig-docs/) is a universal library for training machine learning models, supporting several types of models. Ludwig is Uber's internal library that has been made open source.

Ludwig's main features are:
- no-code approach: the entire training, evaluation and inference process is carried out declaratively
- generalizable: the tool configures itself based on data type declarations, so extending the functionality with new data types is relatively simple
- flexibility: the library allows for very precise control of the training process, while offering sensibly selected default values
- extensibility: adding new models is straightforward and easy

All work with Ludwig comes down to properly preparing a data file (in `csv` format) and preparing one configuration file in `yaml` format (for people not familiar with YAML, I recommend [short tutorial](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started/)). The most interesting idea in Ludwig is to base processing on encoders and decoders that are associated with specific types of data. Ludwig currently [supports the following data types](https://ludwig.ai/latest/configuration/features/supported_data_types/):
- binary
- numerical
- categorical
- sequences
- bag
- set
- text
- time series (timeseries)
- image
- audio
- dates
- vectors
- H3 / geospatial data

The idea of processing in Ludwig is illustrated in the diagram below:

![ludwig structure](image.png)

By combining a specific type of input with a specific type of output, we obtain a specific type of model:
- text + categorical = text classification
- image + categorical = image classification
- image + text = image captioning
- audio + binary = speaker verification
- text + sequence = NER
- categorical, numeric, binary + numeric = regression
- time course + numerical = forecast
- categorical, numeric, binary + binary = e.g. fraud detection


## Text classification

First, we will prepare a dataset containing two types of tweets: tweets related to the COVID-19 pandemic and general tweets. Our task will be to train a model that can recognize tweets about the pandemic.

Run model training using the following command:
```bash
ludwig train --dataset data/tweets/tweets.csv \
    --config_str '{input_features: [{name: tweet, type: text}], output_features: [{name: label, type: category}]}'
```

Of course, specifying training configurations on the command line quickly becomes cumbersome. Create a new file `model-tweets.yaml` and put the following content in it:

```yaml
input_features:
    -
        name: tweet
        type: text

output_features:
    -
        name: label
        type: category

training:
    batch_size: 64
    epochs: 5
```

Start model training using the command:

```bash
ludwig train --dataset data/tweets/tweets.csv --config model-tweets.yaml
```

The default decoder for text data is `parallel_cnn` inspired by the work of Kim [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882). Suppose that instead of using word-level convolution, we try single-character-level convolution. In the `input_features` section, add the `level` key with the value `char` and run the training again.

Perhaps the problem was not really the granularity of the text embedding, but rather the tokenization method (due to the characteristics of tokens found in tweets). Return to the default `word` tokenization level and add a new `preprocessing` dictionary in the `input_features` section, where you can declare that the tokenization must be done by spaces. The fragment of the configuration file should look like this:

```yaml
input_features:
    -
        name: tweet
        type: text
        level: word
        preprocessing:
            word_tokenizer: space
```

Let's add the possibility of modifying the learning parameters. In the `training` section, add two more parameters related to training and change the criterion for stopping training:

```yaml
training:
    batch_size: 64
    epochs: 5
    decay: True
    learning_rate: 0.001
    validation_metric: accuracy
```

We are currently using a convolutional network to encode characters. Instead, we can try to apply a recurrent network. Until recently (i.e. before the emergence of transformer architecture), recurrent networks in the LSTM architecture were commonly used for text processing due to their ability to remember the context of the processed text. Change the `input_features` section of the configuration file to use the RNN LSTM encoder.

```yaml
input_features:
    -
        name: tweet
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        preprocessing:
            word_tokenizer: space
```

Ludwig includes most of the latest language models available in the `huggingface` module. The full list of text encoders is available [here](https://ludwig.ai/latest/configuration/features/text_features/#encoders). Finally, let's see how BERT copes with the task. Change the encoder type to `bert`.

## Working with classic datasets

To illustrate how Ludwig is used for the classic classification problem, we will use the well-known set describing the passengers of the Titanic. View the `data/titanic/train.csv` and `data/titanic/test.csv` datasets.

Then prepare the `model-titanic.yaml` configuration file in the following form.

```yaml
input_features:
    -
        name: Pclass
        type: category
    -
        name: Name
        type: text
    -
        name: Sex
        type: category
    -
        name: Age
        type: numerical
        preprocessing:
          missing_value_strategy: fill_with_mean
    -
        name: SibSp
        type: numerical
    -
        name: Parch
        type: numerical
    -
        name: Ticket
        type: category
    -
        name: Fare
        type: numerical
        preprocessing:
          missing_value_strategy: fill_with_mean
    -
        name: Cabin
        type: category
    -
        name: Embarked
        type: category

output_features:
    -
        name: Survived
        type: binary
```

Run model training by executing the command:

```bash
ludwig train --dataset data/titanic/train.csv --config model-titanic.yaml
```

Let's try to make some slight modifications to the model definition:

- for the `Pclass` attribute, change the encoding method to one-hot (`encoder: sparse`)
- change the `Sex` attribute type to `binary`
- remove information about the port of embarkation

and then run the training again, this time explicitly pointing to where the model is saved:


```bash
ludwig train --dataset data/titanic/train.csv --config model-titanic.yaml --output_directory results/titanic
```

In the next step, we will test the model using the `experiment` command. Before you run the command below, add a training limit of 10 epochs to your configuration file.

```bash
ludwig experiment --k_fold 5 --dataset data/titanic/train.csv --config model-titanic.yaml
```

View the experiment results.

In the next step, we will visualize the learning process.

```bash
ludwig visualize --visualization learning_curves --training_statistics results/titanic/experiment_run/training_statistics.json --output_directory .
```

To get the output `pdf` generated by ludwig, 

```bash
docker cp <container-id>:/home/learning_curves_Survived_accuracy.pdf .
```

## Image classification

To demonstrate how to work with images, we will use a simple shape classification problem. In our data set, we have photos of circles, triangles and squares. Prepare a `model-images.yaml` file with the following model definition:

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn
        preprocessing:
            resize_method: crop_or_pad
            width: 128
            height: 128

output_features:
    -
        name: label
        type: category

training:
    batch_size: 8
    epochs: 25
```

and then run model training:

```bash
ludwig train --dataset image-train.csv --config model-images.yaml --output results/
```

View the result of the learning process (select the appropriate directory with training statistics)

```bash
ludwig visualize --visualization learning_curves --training_statistics results/<run>/training_statistics.json --output_directory .
```

In the next step, we will apply the trained model to a new dataset.

```bash
ludwig predict --dataset image-test.csv --model_path results/<run>/model/
```

Watch the results of applying the model:

```bash
cat results/label_predictions.csv

cat results/label_probabilities.csv
```

Using the command line we can easily combine files and check which examples were misclassified.
```bash
paste image-test.csv results/label_predictions.csv
```

## Serving the model via REST API

Ludwig also provides a simple mechanism through which we can run the model as a service. To do this, we need to install a few dependencies:

```bash
pip install ludwig[serve]
```

and then start the server:

```bash
ludwig serve --model_path results/experiment_run/model --port 8081 --host 0.0.0.0
```

Once the server is started, you can send requests to it from the host:

```bash
curl http://localhost:8081/predict -X POST -F 'image_path=@data/shapes/serve/triangle.png'
```

## Access via Python API

Of course, all Ludwig functionality is also available in Python. The example below shows how this can be done.


```python
from ludwig.api import LudwigModel
import pandas as pd

df = pd.read_csv('data/tweets/tweets.csv')
df.head()

model_definition = {
    'input_features':[
        {'name':'tweet', 'type':'text'},
    ],
    'output_features': [
        {'name': 'label', 'type': 'category'}
    ]
}

model = LudwigModel(model_definition, logging_level=25)
train_stats, _, _ = model.train(dataset=df)

train_stats

tweets = [
    {'tweet': 'I just had my vaccine shot today!'},
    {'tweet': 'Trump claims serious voter fraud in Nevada'},
    {'tweet': 'EU stops the administration of the Pfizer'}
]

output = model.predict(dataset=tweets, data_format='dict')

print(output)
```

## Additional Task

See the description of the dataset containing [tweets about US airlines](https://www.kaggle.com/crowdflower/twitter-airline-sentiment?select=Tweets.csv). This collection is available in the `data/airlines/tweets.csv` directory.

Try training one of the following models yourself:

- sentiment model: predicts the overall sentiment of a tweet (negative, neutral, positive) based on the text of the tweet
- classification model: predicts which airline a given tweet refers to
- regression model: predicts the number of re-tweets a given tweet will get
