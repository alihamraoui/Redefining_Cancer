import pickle as pickle
import pandas as pd
from tqdm import tqdm
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
tqdm.pandas()

with open("clean_text.pkl",'rb') as infile :
      data = pickle.load(infile)

def clean(text):
    sw = stopwords.words("english")
    print('lowercase')
    text = text.progress_apply(lambda x: " ".join(i.lower() for i in x.split()))
    print('remove punctuations')
    text = text.str.replace("[^\w\s]","")
    text = text.str.replace("\d","")
    print('remove stopwords')
    text = text.progress_apply(lambda x: " ".join(i for i in x.split() if i not in sw))
    return text


data["TEXT"] = clean(data["TEXT"])
data.to_pickle("complete_clean_bio.pkl")
