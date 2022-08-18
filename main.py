import requests
import uuid
import pandas as pd
import deepl
from google.cloud import translate_v2 as google_translate
from sacrebleu.metrics import BLEU, CHRF, TER

# Deepl API credentials
deepl_key = ""

# Microsoft API credentials
constructed_url = "https://api.cognitive.microsofttranslator.com/translate"

headers = {
    'Ocp-Apim-Subscription-Key': "",
    'Ocp-Apim-Subscription-Region': "westeurope",
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}


# read in the source WMT sentences from file
def read_in_text(files, num_of_lines):
    file_entries = []
    for ref in files:
        sentences = []
        with open(ref, "r") as f:
            for i in range(num_of_lines):
                line = f.readline()
                sentences.append(line)
        file_entries.append(sentences)

    # return single array of sentences for one file (source file)
    if len(file_entries) == 1:
        return file_entries[0]

    # return array of arrays of sentences (reference files)
    return file_entries


# create a data frame using given variables
def create_data_frame(data, filename):
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(filename)


# translate using given text and translator
def translate(text, translator, source_language, dest_language):
    print("Translating individual sentences...")

    # translate text using specified translator
    if translator == "deepl":
        translations = translate_deepl(text, source_language, dest_language)
    elif translator == "microsoft":
        # preprocess text into dictionary for microsoft format
        dicts = []
        for word in text:
            new_dict = {"text": word}
            dicts.append(new_dict)
        translations = translate_microsoft(dicts, source_language, dest_language)
    else:
        translations = translate_google(text, source_language, dest_language)

    return translations


# translate text as one body, split translation into original sentences
def translate_document(filename, translator, source_language, dest_language, num_of_lines):
    # read in text as a string
    lines = read_in_text(filename, num_of_lines)
    text_to_translate = ''.join(lines)

    # translate text using specified translator
    if translator == "deepl":
        translator = deepl.Translator(deepl_key)
        results_translation = translator.translate_text(text_to_translate, target_lang=dest_language).text
    elif translator == "microsoft":
        body = [{"text": text_to_translate}]
        request = requests.post(constructed_url, params={'api-version': '3.0', 'from': source_language, 'to': [dest_language]}, headers=headers, json=body)
        results = request.json()
        try:
            results_translation = results[0]["translations"][0]["text"]
        except:
            results_translation = ""
    else:
        if dest_language == 'en-us': dest_language = 'en'
        if source_language == 'en-us': source_language = 'en'
        translate_client = google_translate.Client()
        results = translate_client.translate(text_to_translate, source_language=source_language,target_language=dest_language)
        results_translation = results["translatedText"]

    # split translation into array of individual sentences
    document_translations = results_translation.splitlines()

    return document_translations


# send a request to the Microsoft server
def translate_microsoft(source_text, source_language, dest_language):
    print("Calling Microsoft...")
    params = {
        'api-version': '3.0',
        'from': source_language,
        'to': [dest_language]
    }

    request = requests.post(constructed_url, params=params, headers=headers, json=source_text)
    results = request.json()

    translations = []
    for result in results:
        try:
            translations.append(result["translations"][0]["text"])
        except TypeError:
            translations.append("")
        except:
            translations.append("")

    return translations


# send a request to the Deepl server
def translate_deepl(source_text, source_language, dest_language):
    print("Calling Deepl...")
    translator = deepl.Translator(deepl_key)
    results = translator.translate_text(source_text, source_lang=source_language, target_lang=dest_language)

    translations = []
    for result in results:
        translations.append(result.text)

    return translations


# send a request to the Google server
def translate_google(source_text, source_language, dest_language):
    print("Calling Google...")

    if dest_language == 'en-us': dest_language = 'en'
    if source_language == 'en-us': source_language = 'en'

    translate_client = google_translate.Client()
    results = translate_client.translate(source_text, source_language=source_language, target_language=dest_language)

    translations = []
    for result in results:
        translations.append(result.text)

    return translations


# compute translation metrics using SacreBleu
def compute_metrics(reference, candidate):
    print("Reference: ", reference)
    print("Candidate: ", candidate)

    bleu = BLEU()
    bleu_results = bleu.corpus_score(candidate, reference)
    print("BLEU score: ", bleu_results.score)

    ter = TER()
    ter_results = ter.corpus_score(candidate, reference)
    print("TER score: ", ter_results.score)

    chrf = CHRF()
    chrf_results = chrf.corpus_score(candidate, reference)
    print("CHRF score: ", chrf_results.score)

    return bleu_results.score, ter_results.score, chrf_results.score


# compute context translations and metrics for given languages and translator
def compare_contexts(source_file, ref_files, num_of_lines, translator, source_language, dest_language):
    # translate in context
    reference_sentences = read_in_text(ref_files, num_of_lines)
    document_translations = translate_document(source_file, translator, source_language, dest_language, num_of_lines)

    # translate out of context
    sentences_to_translate = read_in_text(source_file, num_of_lines)
    individual_translations = translate(sentences_to_translate, translator, source_language, dest_language)

    # compute metrics
    bleu_in_score, ter_in_score, chrf_in_score = compute_metrics(reference_sentences, document_translations)
    bleu_out_score, ter_out_score, chrf_out_score = compute_metrics(reference_sentences, individual_translations)

    print("BLEU using context: ", bleu_in_score)
    print("BLEU without context: ", bleu_out_score)

    print("Average TER using context: ", ter_in_score)
    print("Average TER without context: ", ter_out_score)

    print("Average CHRF using context: ", chrf_in_score)
    print("Average CHRF without context: ", chrf_out_score)

    return bleu_in_score / 100, bleu_out_score / 100, ter_in_score / 100, ter_out_score / 100, chrf_in_score / 100, chrf_out_score / 100


# create data frame from metrics of all languages and translators
def test_all_data(translator, file_name):
    source_files = [
        ['src-data/newstest2021.en-de.src.en'],
        ['src-data/newstest2021.de-en.src.de'],
        ['src-data/newstest2021.en-cs.src.en'],
        ['src-data/newstest2021.cs-en.src.cs'],
        ['src-data/newstest2021.en-ru.src.en'],
        ['src-data/newstest2021.ru-en.src.ru'],
        ['src-data/newstest2021.en-zh.src.en'],
        ['src-data/newstest2021.zh-en.src.zh'],
        ['src-data/newstest2021.en-ja.src.en'],
        ['src-data/newstest2021.ja-en.src.ja'],
        ['src-data/newstest2021.de-fr.src.de'],
        ['src-data/newstest2021.fr-de.src.fr'],
    ]
    ref_files = [
        ['ref-data/newstest2021.en-de.ref.A.de', 'ref-data/newstest2021.en-de.ref.C.de', 'ref-data/newstest2021.en-de.ref.D.de'],
        ['ref-data/newstest2021.de-en.ref.A.en', 'ref-data/newstest2021.de-en.ref.B.en'],
        ['ref-data/newstest2021.en-cs.ref.A.cs', 'ref-data/newstest2021.en-cs.ref.B.cs'],
        ['ref-data/newstest2021.cs-en.ref.A.en', 'ref-data/newstest2021.cs-en.ref.B.en'],
        ['ref-data/newstest2021.en-ru.ref.A.ru', 'ref-data/newstest2021.en-ru.ref.B.ru'],
        ['ref-data/newstest2021.ru-en.ref.A.en', 'ref-data/newstest2021.ru-en.ref.B.en'],
        ['ref-data/newstest2021.en-zh.ref.A.zh', 'ref-data/newstest2021.en-zh.ref.B.zh'],
        ['ref-data/newstest2021.zh-en.ref.A.en'],
        ['ref-data/newstest2021.en-ja.ref.A.ja'],
        ['ref-data/newstest2021.ja-en.ref.A.en'],
        ['ref-data/newstest2021.de-fr.ref.A.fr'],
        ['ref-data/newstest2021.fr-de.ref.A.de'],
    ]
    source_lang = ['en-us', 'de', 'en-us', 'cs', 'en-us', 'ru', 'en-us', 'zh-cn', 'en-us', 'ja', 'de', 'fr']
    dest_lang = ['de', 'en-us', 'cs', 'en-us', 'ru', 'en-us', 'zh-cn', 'en-us', 'ja', 'en-us', 'fr', 'de']

    # number of lines of text to read in from file
    num_of_lines = 200
    rows = []

    # for each language pair, calculate metrics for in and out of context
    for i in range(len(source_files)):
        print("Round: ", i)
        bleu_in, bleu_out, ter_in, ter_out, chrf_in, chrf_out = compare_contexts(source_files[i], ref_files[i], num_of_lines, translator, source_lang[i], dest_lang[i])
        data_frame_row = {"lang": source_lang[i] + "->" + dest_lang[i], "translator": translator, "BLEU in": round(bleu_in, 3), "TER in": round(ter_in, 3), "CHRF in": round(chrf_in, 3), "BLEU out": round(bleu_out, 3), "TER out": round(ter_out, 3), "CHRF out": round(chrf_out, 3)}
        rows.append(data_frame_row)

    # save metrics to a data frame
    create_data_frame(rows, file_name)


if __name__ == '__main__':
    test_all_data("google", "dataFrames/test_scores_google.csv")
    #test_all_data("microsoft", "dataFrames/test_scores_microsoft.csv")
    #test_all_data("deepl", "dataFrames/test_scores_deepl2.csv")




