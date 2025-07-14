import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import os
import torch
import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer
import re
from typing import Dict, Set, Tuple, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 512
num_epochs = 5

class CustomDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(CustomDecoder, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoded_meanings, hidden_state=None):
        outputs, hidden = self.gru(encoded_meanings, hidden_state)
        logits = self.fc(outputs)
        return logits, hidden

class TranslationSystem:
    def __init__(self, parallel_dir: str = "parallel"):
        self.parallel_dir = parallel_dir

        # Initialize encoder and tokenizer
        self.encoder_model_name = "Helsinki-NLP/opus-mt-en-roa"
        self.encoder = MarianMTModel.from_pretrained(self.encoder_model_name).to(device)
        self.tokenizer = MarianTokenizer.from_pretrained(self.encoder_model_name)
        self.vocab_size = len(self.tokenizer)

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Discover languages and initialize decoders
        self.supported_languages = self._discover_languages()
        self.decoders = self._initialize_decoders()

    def _discover_languages(self) -> Set[str]:
        languages = set()
        if not os.path.exists(self.parallel_dir):
            raise FileNotFoundError(f"Directory {self.parallel_dir} not found")

        pattern = re.compile(r'([a-z]{2})-([a-z]{2})\.txt$')

        for filename in os.listdir(self.parallel_dir):
            match = pattern.match(filename)
            if match:
                lang1, lang2 = match.groups()
                languages.add(lang1)
                languages.add(lang2)

        return languages

    def _initialize_decoders(self) -> Dict[str, CustomDecoder]:
        decoders = {}
        for lang in self.supported_languages:
            decoder = CustomDecoder(hidden_size, self.vocab_size).to(device)
            decoders[lang] = decoder
        return decoders

    def load_decoders(self, directory: str = "decoders") -> None:
        for lang, decoder in self.decoders.items():
            model_path = os.path.join(directory, f"{lang}_decoder.pth")
            if os.path.exists(model_path):
                decoder.load_state_dict(torch.load(model_path, map_location=device))
                decoder.to(device)
                print(f"Loaded {lang} decoder from {model_path}")
            else:
                print(f"No saved decoder found for {lang} in {directory}")

    def get_available_language_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for filename in os.listdir(self.parallel_dir):
            match = re.match(r'([a-z]{2})-([a-z]{2})\.txt$', filename)
            if match:
                lang1, lang2 = match.groups()
                pairs.append((lang1, lang2))
                pairs.append((lang2, lang1))  # Add reverse pair if bidirectional
        return pairs

    def translate(self, input_sentence: str, source_lang: str, target_lang: str) -> str:
        self.encoder.eval()
        self.decoders[target_lang].eval()

        inputs = self.tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            encoder_outputs = self.encoder.model.encoder(**inputs).last_hidden_state

        decoder = self.decoders[target_lang]

        with torch.no_grad():
            logits, _ = decoder(encoder_outputs)

        predicted_tokens = torch.argmax(logits, dim=-1)
        translated_sentence = self.tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)

        return translated_sentence

class TranslationEvaluator:
    def __init__(self, translation_system: TranslationSystem):
        self.translation_system = translation_system
        self.nlp_models = {}
        self.smoothing = SmoothingFunction().method3

    def load_spacy_model(self, lang_code: str):
        if lang_code not in self.nlp_models:
            try:
                model_name = f"{lang_code}_core_news_sm"
                if lang_code == 'en':
                    model_name = "en_core_web_sm"

                self.nlp_models[lang_code] = spacy.load(model_name)
                print(f"Loaded spaCy model for {lang_code}")
            except OSError:
                print(f"Error: Model '{model_name}' not found. "
                      f"Please install it using: python -m spacy download {model_name}")
                raise

        return self.nlp_models[lang_code]

    def tokenize(self, text: str, lang_code: str) -> List[str]:
        nlp = self.load_spacy_model(lang_code)
        doc = nlp(text.lower())
        return [token.text for token in doc if not token.is_space]

    def parse_file(self, file_path: str) -> List[Tuple[str, str]]:
        pairs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for i in range(0, len(lines), 2):
                    if i + 1 < len(lines):
                        source = lines[i].strip()
                        target = lines[i + 1].strip()
                        if source and target:
                            pairs.append((source, target))
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
        return pairs

    def calculate_metrics(self, file_path: str, source_lang: str, target_lang: str) -> Tuple[float, float]:
        pairs = self.parse_file(file_path)
        if not pairs:
            return 0.0, 0.0

        bleu_scores = []
        meteor_scores = []

        for source_sentence, reference_sentence in pairs:
            try:
                generated_sentence = self.translation_system.translate(source_sentence, source_lang, target_lang)
                reference_tokens = self.tokenize(reference_sentence, target_lang)
                generated_tokens = self.tokenize(generated_sentence, target_lang)

                if not reference_tokens or not generated_tokens:
                    continue

                bleu = sentence_bleu(
                    [reference_tokens],
                    generated_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=self.smoothing
                )
                bleu_scores.append(bleu)

                meteor = meteor_score([reference_tokens], generated_tokens)
                meteor_scores.append(meteor)

            except Exception as e:
                print(f"Error processing sentence pair: {str(e)}")
                continue

        if not bleu_scores or not meteor_scores:
            return 0.0, 0.0

        return sum(bleu_scores) / len(bleu_scores), sum(meteor_scores) / len(meteor_scores)

    def evaluate_all_pairs(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        results = {}
        language_pairs = self.translation_system.get_available_language_pairs()

        for source_lang, target_lang in language_pairs:
            file_path = os.path.join(
                self.translation_system.parallel_dir,
                f"{min(source_lang, target_lang)}-{max(source_lang, target_lang)}.txt"
            )

            try:
                avg_bleu, avg_meteor = self.calculate_metrics(file_path, source_lang, target_lang)
                results[(source_lang, target_lang)] = (avg_bleu, avg_meteor)
            except Exception as e:
                print(f"Error processing language pair {source_lang}->{target_lang}: {str(e)}")
                results[(source_lang, target_lang)] = (0.0, 0.0)

        return results

def main():
    translation_system = TranslationSystem()
    print(f"Discovered languages: {translation_system.supported_languages}")
    print(f"Available language pairs: {translation_system.get_available_language_pairs()}")

    translation_system.load_decoders()

    evaluator = TranslationEvaluator(translation_system)
    results = evaluator.evaluate_all_pairs()

    print("\nFinal Results Summary:")
    for (source, target), (bleu, meteor) in results.items():
        print(f"{source} -> {target}:")
        print(f"  BLEU = {bleu:.4f}")
        print(f"  METEOR = {meteor:.4f}")

if __name__ == "__main__":
    main()