import spacy 
from spacy.lang.en import STOP_WORDS
nlp = spacy.load('en_core_web_sm')

#pipeline przetwarzania
def preprocessing_text(text): 
    text = text.strip().lower()
    doc = nlp(text.lower())
    tokens = [token for token in doc if token.is_alpha] #tokenizacja s
    lemmas = [token.lemma_ for token in tokens] #lematyzacja
    no_stop = [word for word in lemmas if word not in STOP_WORDS] #lista stop
    no_single = [word for word in no_stop if len(word)>1]
    return " ".join(no_single)