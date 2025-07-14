# Unsupervised Neural Machine Translation using Monolingual Corpora

This project demonstrates an end-to-end unsupervised machine translation system that requires **no parallel data** for training. It leverages a **frozen pretrained encoder** (MarianMT) and trains **custom GRU decoders per language** using only **monolingual corpora**.

This approach is highly scalable and ideal for **low-resource languages**, where parallel data is limited or unavailable.

---

## Key Highlights

- No parallel corpora required — only monolingual data
- Custom decoders per language (English, French, Spanish) trained using **backtranslation**
- Measured improvement via BLEU, METEOR, TER scores on real-world samples
- [**Report Attached**](https://github.com/granth23/Machine-Translation/blob/main/Report_Unsupervised_Neural_Machine_Translation.pdf) — includes implementation details, metrics, and future work

---

## System Architecture

### How It Works (Backtranslation Training)

1. **Input text** (e.g., in Spanish) is passed into a **frozen pretrained MarianMT encoder**.
2. This is converted into a **shared context vector** (language-agnostic semantic representation).
3. The corresponding **language-specific decoder** (e.g., Spanish decoder) attempts to **reconstruct the original input**.
4. The **output text** is compared to the original input to compute **loss**, which is used to update only the decoder weights.
5. This enables **unsupervised training**, simulating supervised learning with monolingual inputs.

<img width="1600" height="1014" alt="image" src="https://github.com/user-attachments/assets/37e9c1c5-0e23-415b-9403-e897dda6fd64" />


---

## Evaluation Results

We tested our models on the **Tatoeba parallel corpus** and evaluated using BLEU, METEOR, and TER metrics.

| Language Pair | BLEU   | METEOR | Comments                    |
|---------------|--------|--------|-----------------------------|
| EN → ES       | 0.0809 | 0.1097 | Stronger for Romance pairs |
| FR → ES       | 0.0957 | 0.1525 | Highest performance         |
| ES → EN       | 0.0739 | 0.1105 | Consistent across directions|


---

## Technologies Used

<p align="left">
  <!-- Python -->
  <a href="https://www.python.org" target="_blank" rel="noreferrer" style="margin: 10px;">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" width="40" height="40"/>
  </a>

  <!-- NumPy -->
  <a href="https://numpy.org/" target="_blank" rel="noreferrer" style="margin: 10px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" alt="NumPy" width="40" height="40"/>
  </a>

  <!-- TensorFlow -->
  <a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer" style="margin: 10px;">
    <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="TensorFlow" width="40" height="40"/>
  </a>

  <!-- Hugging Face Transformers -->
  <a href="https://huggingface.co/transformers/" target="_blank" rel="noreferrer" style="margin: 10px;">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="40" height="40"/>
  </a>
</p>

- Python 
- TensorFlow & Keras
- NumPy
- HuggingFace Transformers (MarianMT)
- Tatoeba Corpus for evaluation



