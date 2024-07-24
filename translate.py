from googletrans import Translator
import sys

with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
    stop_words = [line.strip() for line in f if line.strip()]
    f.close()
def translate_vietnamese_to_english(text):
    translator = Translator()
    translated = translator.translate(text, src='vi', dest='en').text
    filtered_words = [word for word in translated if word.lower() not in stop_words]
    result = ''.join(filtered_words)
    return result
