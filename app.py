import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import MarianMTModel, MarianTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 512
num_epochs = 5

encoder_model_name = "Helsinki-NLP/opus-mt-en-roa"
encoder = MarianMTModel.from_pretrained(encoder_model_name).to(device)
tokenizer = MarianTokenizer.from_pretrained(encoder_model_name)

for param in encoder.parameters():
    param.requires_grad = False

class CustomDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(CustomDecoder, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoded_meanings, hidden_state=None):
        outputs, hidden = self.gru(encoded_meanings, hidden_state)
        logits = self.fc(outputs)
        return logits, hidden

vocab_size = len(tokenizer)
criterion = nn.CrossEntropyLoss()

def train_step(batch_sentences, source_lang):
    encoder.eval()
    decoders[source_lang].train()

    inputs = tokenizer(
        batch_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    targets = inputs

    with torch.no_grad():
        encoder_outputs = encoder.model.encoder(**inputs).last_hidden_state

    decoder = decoders[source_lang]
    logits, _ = decoder(encoder_outputs)

    target_tokens = targets["input_ids"].reshape(-1)
    logits = logits.reshape(-1, logits.size(-1))

    loss = criterion(logits, target_tokens)

    optimizers[source_lang].zero_grad()
    loss.backward()
    optimizers[source_lang].step()

    return loss.item()

def translate(input_sentence, source_lang, target_lang):
    encoder.eval()
    decoders[target_lang].eval()

    inputs = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        encoder_outputs = encoder.model.encoder(**inputs).last_hidden_state

    decoder = decoders[target_lang]

    with torch.no_grad():
        logits, _ = decoder(encoder_outputs)

    predicted_tokens = torch.argmax(logits, dim=-1)
    translated_sentence = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)

    return translated_sentence

def save_decoders(decoders, directory="decoders"):
    os.makedirs(directory, exist_ok=True)
    for lang, decoder in decoders.items():
        model_path = os.path.join(directory, f"{lang}_decoder.pth")
        torch.save(decoder.state_dict(), model_path)
        print(f"Saved {lang} decoder to {model_path}")

def load_decoders(decoders, directory="decoders"):
    for lang, decoder in decoders.items():
        model_path = os.path.join(directory, f"{lang}_decoder.pth")
        if os.path.exists(model_path):
            decoder.load_state_dict(torch.load(model_path, map_location=device))
            decoder.to(device)
            print(f"Loaded {lang} decoder from {model_path}")
        else:
            print(f"No saved decoder found for {lang} in {directory}")

def create_batches(sentences, batch_size=32):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i:i + batch_size]

def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file.readlines()]
    return sentences

method = int(input("Enter 0 if you want to create a new model else enter 1: "))

if method == 0:
    total_languages = int(input("Enter Total Languages: "))

    corpora = {}
    decoders = {}
    optimizers = {}

    for i in range(total_languages):
        language = input("Enter language initials: ")
        source = input("Enter corpora path: ")
        corpora[language] = source
        decoders[language] = CustomDecoder(hidden_size, vocab_size).to(device)
        optimizers[language] = optim.Adam(decoders[language].parameters(), lr=1e-4)


    for key, val in corpora.items():
        source_lang = key
        file_path = val
        sentences = read_sentences(file_path)[:100]
        batch_size = 32
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in create_batches(sentences, batch_size):
                loss = train_step(batch, source_lang)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    while True:
        source_lang = input("Enter source language for translation: ")
        target_lang = input("Enter target language for translation: ")
        input_sentence = input("Enter sentence for translation: ")
        translated_sentence = translate(input_sentence, source_lang, target_lang)
        print(f"Translated Sentence: {translated_sentence}")
        cont = int(input("If you want to continue press 1 else press 0: "))
        if cont == 1:
            continue
        else:
            break
    save = int(input("Enter 1 if you would like to save the model else enter 0: "))
    if save == 1:
        save_decoders(decoders)

else:
    decoders = {
        "en": CustomDecoder(hidden_size, vocab_size).to(device),
        "fr": CustomDecoder(hidden_size, vocab_size).to(device),
        "es": CustomDecoder(hidden_size, vocab_size).to(device),
    }

    load_decoders(decoders)
    while True:
        source_lang = input("Enter source language for translation: ")
        target_lang = input("Enter target language for translation: ")
        input_sentence = input("Enter sentence for translation: ")
        translated_sentence = translate(input_sentence, source_lang, target_lang)
        print(f"Translated Sentence: {translated_sentence}")
        cont = int(input("If you want to continue press 1 else press 0: "))
        if cont == 1:
            continue
        else:
            break

print("Thank You")