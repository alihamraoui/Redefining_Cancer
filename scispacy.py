import pandas as pd
import spacy
import pickle
from spacy.lang.en import English
import en_core_sci_md
from scispacy.abbreviation import AbbreviationDetector
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import nltk
import string
import re
from pandarallel import pandarallel
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 
from tqdm import tqdm
tqdm.pandas()


def process_Text(text):
    wordlist=[]
    doc = nlp(text)
    for ent in doc.ents:
        wordlist.append(ent.text)
    return ' '.join(wordlist)  


def clean_text(text ): 
    text = text.translate(str.maketrans('', '', string.punctuation))
    text1 = ''.join([w for w in text if not w.isdigit()]) 
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    
    text2 = text1.lower()
    text2 = REPLACE_BY_SPACE_RE.sub('', text2) # replace REPLACE_BY_SPACE_RE symbols by space in text
    return text2


def lemmatize_text(text):
    wordlist=[]
    lemmatizer = WordNetLemmatizer() 
    sentences=sent_tokenize(text)
    
    intial_sentences= sentences[0:1]
    final_sentences = sentences[len(sentences)-2: len(sentences)-1]
    
    for sentence in intial_sentences:
        words=word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))
    for sentence in final_sentences:
        words=word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))       
    return ' '.join(wordlist) 


def get_lang_detector(nlp, name):
    return LanguageDetector()


df_tab = pd.read_csv('training_variants.zip')
df_text = pd.read_csv("training_text.zip",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
df_merge = pd.merge(df_tab,df_text, on="ID",how="left")
df_merge.loc[df_merge["TEXT"].isnull(),"TEXT"] = df_merge["Gene"] + " " + df_merge["Variation"]

#spacy
nlp = spacy.load("en_core_sci_md")
nlp.add_pipe("abbreviation_detector")


Language.factory("language_detector", func=get_lang_detector)
pandarallel.initialize(progress_bar=True, nb_workers = 10)
df_merge['TEXT'] = df_merge['TEXT'][:100].parallel_apply(process_Text)

df_merge.to_pickle("clean_text.pkl")

