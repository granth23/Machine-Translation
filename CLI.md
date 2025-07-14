# Neural Machine Translation CLI

This is a command-line interface for a neural machine translation system.

## Load Dataset and Weights

1. Monolingual Corpora: If retraining:
   - Download the corpora from the [drive link](https://drive.google.com/drive/folders/1Uu6_HB_GOnrR0mrosmXn4XZ8mK-S1mcw?usp=sharing) (folder name: `dataset`) or use your own
   - Share the path of the corpora.
2. Weights: If using pretrained weights:
   - Download from the [drive link](https://drive.google.com/drive/folders/1Uu6_HB_GOnrR0mrosmXn4XZ8mK-S1mcw?usp=sharing) (folder name: `decoders`)
   - Put the weights in a folder named `decoders`
   - Make sure the folder is in the same directory as the app.py
3. Multilingual Corpora: For BLEU and METEOR scores:
   - Download the files from the [drive link](https://drive.google.com/drive/folders/1Uu6_HB_GOnrR0mrosmXn4XZ8mK-S1mcw?usp=sharing) (folder name: `parallel`)
   - Put the files in a folder named `parallel`
   - Make sure the folder is in the same directory as the app.py

### Notes

- The resources in the drive link are for english, french and spanish.

## Usage of app.py

When you run the program, you will be prompted with two options:

### Option 0: Create and Train New Model

1. Enter `0` when prompted to create a new model
2. Specify the number of languages you want to support
3. For each language:
   - Enter the language initial (e.g., "en", "fr", "es")
   - Provide the path to your training corpus
4. The model will train automatically
5. After training, you can:
   - Enter source language
   - Enter target language
   - Input your sentence for translation
   - Choose to continue translating (1) or stop (0)
6. Finally, choose to save the model (1) or discard it (0)

#### Notes

- Change number of epochs as per usage
- Change batch size as per usage

### Option 1: Use Pre-trained Model

1. Enter `1` when prompted to use existing model
2. You can then:
   - Enter source language
   - Enter target language
   - Input your sentence for translation
   - Choose to continue translating (1) or stop (0)

## Usage of scores.py

The system evaluates translation quality using two metrics. Make sure the decoders are stored prior to evaluation. Install the necessary spacy tokenizer for the languages used. Make sure the multilingual datasets are present for all possible language pairs as in the decoders.

### BLEU Score

- Ranges from 0 to 1
- Higher scores indicate better translations
- Measures how similar the machine translation is to professional human translations

### METEOR Score

- Ranges from 0 to 1
- Higher scores indicate better translations
- Considers synonyms and paraphrases when evaluating translation quality

### Test data format

- We have used parallel corpus from Tatoeba for testing that maps english-
  spanish, spanish-french, and english-french.
- The format of the test data is such that the first line is in the first language, and the second line is the equivalent sentence in the second language.
