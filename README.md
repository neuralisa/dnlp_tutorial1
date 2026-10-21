# PyTorch for NLP

1. [MNIST](01-mnist.ipynb)
2. [GloVe text classification](02-headlines.ipynb)
3. [Build the NER model](03-ner.ipynb)
4. [Evaluate and submit the checkpoint](04-ner-evaluation.ipynb)
5. [RNN tagging](05-rnn-ner.ipynb)
6. [RNN and bidirectional RNN](06-rnn-birnn.ipynb)

## Run

```sh
git clone https://github.com/neuralisa/dnlp_tutorial1.git
cd dnlp_tutorial1
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`.
Open a notebook and run its cells in order. CPU is sufficient.
The MNIST sample and headline GloVe vectors are included.

Notebooks 4–6 need the course-provided files in:

```text
data/conll2003/train.txt
data/conll2003/valid.txt
data/conll2003/test.txt
```

Keep the supplied NER utility and evaluation scripts beside the notebooks.
The RNN notebook trains 32-dimensional embeddings by default; placing `glove.6B.100d.txt` in this folder enables its GloVe option.

[Interactive session](https://neuralisa.com/dnlp/ner-lab/tutorials.html) · [Submit a checkpoint](https://neuralisa.com/ner-challenge/)

## Data and code

MNIST: 128 training images and 32 test images, with source checksums in `data/mnist-source.json`.
GloVe: included 100-dimensional rows from [Stanford GloVe 6B](https://nlp.stanford.edu/projects/glove/), released under PDDL. Headlines and the small NER examples are authored teaching data.
The checkpoint utilities and evaluator come from [the course repository](https://github.com/neuralisa/tuw-dnlp-ner-2026s/tree/a4a5e1118d6abbee70a6c95203f4811cdd193770).
