# importing required libraries
import subprocess
import sys
print("installing ftfy python package...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "ftfy"])
print("Done!")
from ftfy import fix_text
import re
import html
from string import punctuation
import nltk
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import warnings
warnings.filterwarnings('ignore')


# fixing unicode characters
def fix_unicode_characters(text):
    '''
    Fixes unicode characters using ftfy python package
    text: text to clean
    '''
    return fix_text(text)


# defining functions for cleaning
punctuations = list(punctuation)
stopwords = stopwords.words('english')
word_lemetizer = WordNetLemmatizer()

# Converts nltk tag to wordnet tag since we will lemmatize based in the POS
# This function will be used in the below function
def nltk_tag_to_wordnet_tag(nltk_tag):
  if nltk_tag.startswith('J'):
      return wordnet.ADJ
  elif nltk_tag.startswith('V'):
      return wordnet.VERB
  elif nltk_tag.startswith('N'):
      return wordnet.NOUN
  elif nltk_tag.startswith('R'):
      return wordnet.ADV
  else:          
      return "UNKNOWN"


# Custom POS based lemmatizer. If a corresponding wordnet tag is found, it is lemmatized accordingly,
# else we'll append the word as is
# This function will be used in the below function
def pos_based_lematizer(sentence_tokens):
  tag_word_pair = nltk.pos_tag(sentence_tokens)
  final_tokens = []
  for pair in tag_word_pair:
    wrd,tg = pair[0], nltk_tag_to_wordnet_tag(pair[1]) #nltk pos tag to wordnet pos tag conversion - conversion function defined above
    if tg=="UNKNOWN":
        final_tokens.append(wrd)
    else:
        final_tokens.append(word_lemetizer.lemmatize(wrd, tg))
  
  return final_tokens



# Bringing it all together. Each step is commented below
def make_tokens(text):
  '''
  This function will remove punctuations, lowercase the word, escape the html entities, remove email ids and 
  corresponding jargons, replaces hyperlinks and ip addresses with corresponding words, removes date and 
  time, replace contractions, remove special characters, remove stopwords, lemmatize and tokenize the text
  
  text: text to clean, lemmatize and tokenize (string)
  
  Returns cleaned, lemmatized and tokenized text
  '''

  #replacing ip address with the word 'ip address'
  text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}[^0-9]",' ip address ',text) 
  #removing date and time formats
  text = re.sub(r"(\d+/\d+/\d+)",' ',text) 
  text = re.sub(r"(\d+-\d+-\d+)",' ',text) 
  text = re.sub(r"(\d+:\d+:\d+)",' ',text) 
  #replacing hyperlinks and other weblinks with the word "hyperlink"
  text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",' hyperlink ',text)
  #removing email id's
  text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[(com|org|edu)]{2,4}",' ',text)
  # removing html entities eg: &amp;, &gt;, etc.
  text = html.unescape(text) 

  word = text.split()
  word = [w.lower() for w in word if w not in punctuations] #removing punctuations and lowercasing
  word = " ".join(word)

  # replacing contractions
  word = re.sub(r"what's", "what is ", word)
  word = re.sub(r"\'s", " ", word)
  word = re.sub(r"\'ve", " have ", word)
  word = re.sub(r"\'re", " are ", word)
  word = re.sub(r"\'r", " your ", word)
  word = re.sub(r"can't", "cannot ", word)
  word = re.sub(r'won\'t', " will not ", word)
  word = re.sub(r"don't", "do not ", word)
  word = re.sub(r"mustn't", "must not ", word)
  word = re.sub(r"shouldn't", "should not ", word)
  word = re.sub(r"haven't", "have not ", word)
  word = re.sub(r"i'm", "i am ", word)
  word = re.sub(r"\bi m\b", "i am ", word)
  word = re.sub(r"\bok\b", "okay", word)
  word = re.sub(r"\'re", " are ", word)
  word = re.sub(r"\'d", " would ", word)
  word = re.sub(r"\'ll", " will ", word)
  word = re.sub(r'didn\'t', " did not ", word)
  word = re.sub(r'didnt', " did not ", word)
  word = re.sub(r'ain\'t', " is not ", word)

  # removing email related jargons
  word = re.sub(r"received from:",' ',word)
  word = re.sub(r"from:",' ',word)
  word = re.sub(r"to:",' ',word)
  word = re.sub(r"subject:",' ',word)
  word = re.sub(r"sent:",' ',word)
  word = re.sub(r"ic:",' ',word)
  word = re.sub(r"cc:",' ',word)
  word = re.sub(r"bcc:",' ',word)
  word = re.sub(r"\bhi\b",' ',word)
  word = re.sub(r"hello",' ',word)
  word = re.sub(r"gmail",' ',word)
  word = re.sub(r"greetings",' ',word)
  word = re.sub(r"e - mail", "email", word)

  word = re.sub(r"[^A-Za-z0-9]", " ", word) #removing anything that is not a letter from a-z
  word = re.sub(r"\s{2,}", " ", word) #removing more than >=2 spaces from text

  tokens = RegexpTokenizer('[A-Za-z0-9]+').tokenize(word) #converting to tokens
  tokens = pos_based_lematizer(tokens) #lematizing - custom lemmatizer function defined above
  tokens = [t for t in tokens if t not in stopwords] #removing stopwords
  cleaned_text = ' '.join(tokens)
  return cleaned_text