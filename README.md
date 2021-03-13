<h1 align="center">encode-attend-navigate-pytorch</h1>
<p align="center">
Pytorch implementation of [encode-attend-navigate](https://github.com/MichelDeudon/encode-attend-navigate), a Deep Reinforcement Learning based TSP solver.
</p>

**⚠️ I couldn't reach the same results so far. The TF implementation reach score ~0.1 as good as the 2-opt solution. But with this implementation, the difference between only-RL and RL + 2opt is almost 1 point of difference. I'm currently doing hyper-parameters search, but any help is welcomed :)**

## Get started

### Run on Colab

You can leverage the free GPU on Colab to train this model. Just run this notebook :
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

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
python main.py att_hidden=128 crit_hidden=512 enc_stacks=1 ff_hidden=256 lr=0.0002 lr_decay_rate=0.94 lr_decay_steps=5000 p_dropout=0.1 query_hidden=128 steps=25000
```

On Colab, with a `Tesla T4` GPU. It tooks 1h 4m for the training to complete.

Here is the training curves :

![W B Chart 3_13_2021, 11_53_02 AM](https://user-images.githubusercontent.com/43774355/111016726-d4c5f700-83f2-11eb-9c28-d91acda7eacc.png)
![W B Chart 3_13_2021, 11_53_14 AM](https://user-images.githubusercontent.com/43774355/111016728-d5f72400-83f2-11eb-880b-9258bacf33d2.png)

![W B Chart 3_13_2021, 11_53_30 AM](https://user-images.githubusercontent.com/43774355/111016730-d68fba80-83f2-11eb-811c-ec861d168a18.png)
![W B Chart 3_13_2021, 11_53_42 AM](https://user-images.githubusercontent.com/43774355/111016731-d7c0e780-83f2-11eb-89e5-27077e03edc4.png)

## Implementation

This code is a direct translation of the [official TF 1.x implementation](https://github.com/MichelDeudon/encode-attend-navigate), by @MichelDeudon.

Please refer to their README for additional details.

---

To ensure the Pytorch implementation produces the same results as the original implementation, I compared the outputs of each layer given the same inputs and check if they are the same.

You can find (and run) these tests on this Colab notebook :

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HChapUUC_3cZoZsG1A3WJLwclQRsyuR2?usp=sharing)