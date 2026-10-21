# PyTorch for NLP

Start with [MNIST](01-mnist.ipynb) and [GloVe text classification](02-headlines.ipynb), then work through [building the NER model](03-ner.ipynb) and [evaluation and submission](04-ner-evaluation.ipynb), before trying [RNN tagging](05-rnn-ner.ipynb) and the [bidirectional RNN comparison](06-rnn-birnn.ipynb).

```sh
git clone https://github.com/neuralisa/dnlp_tutorial1.git
cd dnlp_tutorial1
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

On Windows, activate the environment with `.venv\Scripts\activate`, then open a notebook in Jupyter and run its cells in order; all six notebooks can run on CPU.

The MNIST sample and headline GloVe vectors are included, while notebooks 4–6 also need the course-provided `train.txt`, `valid.txt` and `test.txt` files in `data/conll2003/`, with the supplied utility scripts kept beside the notebooks.

The included vectors come from [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) under PDDL, the NER utilities come from the [course repository](https://github.com/neuralisa/tuw-dnlp-ner-2026s), and MNIST source checksums are in `data/mnist-source.json`.

[Interactive session](https://neuralisa.com/dnlp/ner-lab/tutorials.html) · [Submit a checkpoint](https://neuralisa.com/ner-challenge/)
