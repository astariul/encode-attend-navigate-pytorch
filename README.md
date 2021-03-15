<h1 align="center">encode-attend-navigate-pytorch</h1>
<p align="center">
Pytorch implementation of <a href="https://github.com/MichelDeudon/encode-attend-navigate">encode-attend-navigate</a>, a Deep Reinforcement Learning based TSP solver.
</p>

## Get started

### Run on Colab

You can leverage the free GPU on Colab to train this model. Just run this notebook :
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tggr-QIQSyt7jnjZRuBp5wBt6eDoC1-c?usp=sharing)

### Run locally

Clone the repository :

```console
git clone https://github.com/astariul/encode-attend-navigate-pytorch.git
cd encode-attend-navigate-pytorch
```

---

Install dependencies :

```console
pip install -r requirements.txt
```

---

Run the code :

```console
python main.py
```

---

You can specify your own configuration file :

```console
python main.py config=my_conf.yaml
```

---

Or directly modify parameters from the command line :

```console
python main.py lr=0.002 max_len=100 batch_size=64
```

### Expected results

I ran the code with the following command line :

```console
python main.py enc_stacks=1 lr=0.0002 p_dropout=0.1
```

On Colab, with a `Tesla T4` GPU, it tooks 1h 43m for the training to complete.

Here is the training curves :

<img src="https://user-images.githubusercontent.com/43774355/111126924-9416db00-85b6-11eb-932e-1af504d91d5a.png" width="400"> <img src="https://user-images.githubusercontent.com/43774355/111126920-937e4480-85b6-11eb-818e-4cd3dd93e94e.png" width="400">

<img src="https://user-images.githubusercontent.com/43774355/111126917-92e5ae00-85b6-11eb-8e91-116503e345ac.png" width="400"> <img src="https://user-images.githubusercontent.com/43774355/111126909-911bea80-85b6-11eb-8337-0c3e87bfdd21.png" width="400">

---

After training, here is a few example of path generated :

<img src="https://user-images.githubusercontent.com/43774355/111127122-cc1e1e00-85b6-11eb-86f1-0aa93f1b9e13.png" width="400"> <img src="https://user-images.githubusercontent.com/43774355/111127120-cb858780-85b6-11eb-8ac9-597325f886d5.png" width="400">

<img src="https://user-images.githubusercontent.com/43774355/111127116-caecf100-85b6-11eb-92a3-85ca4f178333.png" width="400"> <img src="https://user-images.githubusercontent.com/43774355/111127110-ca545a80-85b6-11eb-8e45-c0ccd00c455d.png" width="400">

🔎 _As you can see, the difference of score between "RL only" and "RL + 2-opt" is more important than the original repository. I'm still trying to find the issue._

## Implementation

This code is a direct translation of the [official TF 1.x implementation](https://github.com/MichelDeudon/encode-attend-navigate), by @MichelDeudon.

Please refer to their README for additional details.

---

To ensure the Pytorch implementation produces the same results as the original implementation, I compared the outputs of each layer given the same inputs and check if they are the same.

You can find (and run) these tests on this Colab notebook : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HChapUUC_3cZoZsG1A3WJLwclQRsyuR2?usp=sharing)