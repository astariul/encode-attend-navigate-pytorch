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

On Colab, with a `Tesla T4` GPU, it tooks 1h 46m for the training to complete.

Here is the training curves :

<img src="https://user-images.githubusercontent.com/43774355/116061592-03a3de00-a6be-11eb-8d7a-71dce9586c3e.png" width="400"> <img src="https://user-images.githubusercontent.com/43774355/116061604-07376500-a6be-11eb-8368-9d9d87e277cb.png" width="400">

<img src="https://user-images.githubusercontent.com/43774355/116061618-0b638280-a6be-11eb-86a6-40e61cb84b0c.png" width="400"> <img src="https://user-images.githubusercontent.com/43774355/116061630-0e5e7300-a6be-11eb-8313-672da3965174.png" width="400">

---

After training, here is a few example of path generated :

<img src="https://user-images.githubusercontent.com/43774355/116062078-8a58bb00-a6be-11eb-86ff-3de0e6735950.png" width="400"> <img src="https://user-images.githubusercontent.com/43774355/116062074-89c02480-a6be-11eb-8668-e8141e71e5aa.png" width="400">

<img src="https://user-images.githubusercontent.com/43774355/116062069-888ef780-a6be-11eb-96f8-183cd65c8af5.png" width="400"> <img src="https://user-images.githubusercontent.com/43774355/116062082-8a58bb00-a6be-11eb-857d-df51d7aa7763.png" width="400">

## Implementation

This code is a direct translation of the [official TF 1.x implementation](https://github.com/MichelDeudon/encode-attend-navigate), by @MichelDeudon.

Please refer to their README for additional details.

---

To ensure the Pytorch implementation produces the same results as the original implementation, I compared the outputs of each layer given the same inputs and check if they are the same.

You can find (and run) these tests on this Colab notebook : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HChapUUC_3cZoZsG1A3WJLwclQRsyuR2?usp=sharing)