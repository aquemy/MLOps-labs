# Data annotation with Prodigy

## Recognize named entities (NER)

1. Prodigy is a tool strongly related to `spacy` library. It defines named entity recognition patterns in a analogous way to the matcher mechanism present in `spacy`.

2. Normally we would start by building a file that contains some examples of what a programming language name looks like. For this we will use the format popularized by the `spacy` library.

```
{"label": "PROG_LANG", "pattern": [{"lower": "java"}]}
{"label": "PROG_LANG", "pattern": [{"lower": "c"}, {"lower": "+"}, {"lower": "+"}]}
{"label": "PROG_LANG", "pattern": [{"lower": "objective"}, {"lower": "c"}]}
...
```
3. However, preparing such a file is time-consuming. Instead, we'll let Prodigy build this file for us from a small set of seed terms.

```bash
prodigy terms.teach language_names en_core_web_lg --seeds python,julia,prolog,lisp,java,smalltalk,go
```

4. After generating the appropriate number of examples, we can see the result of this annotation

```bash
prodigy db-out language_names
```

5. This dataset should still be translated into a SpaCy compatible format

```bash
prodigy terms.to-patterns language_names --label PROG_LANG --spacy-model blank:en > ./language_names.jsonl

head ./language_names.jsonl
```

6. With the file prepared in this way, we can start manual annotation of data from comments.

```bash
prodigy ner.manual programming_languages en_core_web_lg ./programming.jsonl.bz2 --loader reddit --label PROG_LANG --patterns language_names.jsonl
```

7. After annotating an appropriate number of cases, we can view the annotations

```bash
prodigy print-dataset programming_languages
```

8. It is also possible to export the annotated data set

```bash
prodigy db-out programming_languages
```

9. The next step is to train the initial NER model. This is just the beginning, we will improve the quality of this model later.

```bash
prodigy train /tmp/initial --ner programming_languages --base-model en_core_web_lg --eval-split 0.2 --training.eval_frequency 100 --training.patience 1000
```

10. In the next step, we will check how well our NER model performs. We will ask the model for annotation and we will only assess the model's decision in a binary way, which is of course a much easier and less tiring job. We will also exclude from the annotation cases that we have previously noted.

```bash
prodigy ner.correct programming_languages_corrected /tmp/initial/model-best ./programming.jsonl.bz2 --loader reddit --label PROG_LANG --exclude programming_languages
```

11. The final step is to combine both datasets and train the final entity recognition model

```bash
prodigy train /tmp/final --ner programming_languages,programming_languages_corrected --base-model en_core_web_lg --eval-split 0.2 --training.eval_frequency 100 --training.patience 1000
```

12. Let's try out the model we created

```python
import spacy

nlp = spacy.load('/tmp/final/model-last')

doc = nlp('My favourite programming languages are Python, C++ and Scheme')

for e in doc.ents:
    print(e.text, e.label_, e.start, e.end)
```

## Image annotation

1. Now let's try to use Prodigy to manually mark the elements of interest in facial photos.

```bash
prodigy image.manual faces_dataset ./images --label MOUTH,EYES
```

2. We can also run the photo classification process. In the simplest version, we annotate photos in a binary way, answering a simple question: does the photo show an adult?

```bash
prodigy mark adult_child_image ./images --loader  images --label ADULT --view-id classification
```

3. If we want to use Prodigy to classify photos when the number of classes is greater than 2, we must prepare our own "recipe" using the `choice` interface. It comes down to decorating a function, which is a generator that returns appropriately formatted dictionaries.

```python
import prodigy
from prodigy.components.loaders import Images

OPTIONS = [
    {"id": 1, "text": "SERIOUS"},
    {"id": 2, "text": "SAD"},
    {"id": 3, "text": "GLAD"},
]

@prodigy.recipe("classify-images")
def classify_images(dataset, source):
    def get_stream():
        stream = Images(source)
        for example in stream:
            example["options"] = OPTIONS
            yield example

    return {
        "dataset": dataset,
        "stream": get_stream(),
        "view_id": "choice",
        "config": {
            "choice_style": "single",
            "choice_auto_accept": True
        }
    }
```

```bash
prodigy classify-images emotions_dataset ./images -F recipe.py
```

## Text classification

Text classification is very similar to training the NER model, with the difference that the entire document (or its sentences) is evaluated.

1. In the first step, we need to manually annotate a certain number of comments so that the model can find the relationship between the words appearing in the text and the text label.

```bash
prodigy textcat.manual programming_comment programming.jsonl.bz2 --loader reddit --label PROGRAMMING,OTHER --exclusive
```

```bash
prodigy print-dataset programming_comment
```

2. The second step is to run model training on the prepared annotations

```bash
prodigy train /tmp/textcat --textcat programming_comment --training.eval_frequency 100 --training.patience 1000
```

3. We can easily check the trained model in action

```python
import spacy

nlp = spacy.load('/tmp/textcat/model-last')

doc = nlp('Java has just released version 15')

for cat in doc.cats:
    print(f'label {cat}: {doc.cats[cat]}')
```

## Task

Use the `homebrewing.jsonl.bz2` file to train your own NER model to recognize beer types (APA, IPA, Vermont, pilsner, lager, weizen, bock, helles, ...)
