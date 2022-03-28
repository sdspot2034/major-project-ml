import numpy as np
import pandas as pd
import spacy
from spellchecker import SpellChecker
from nltk.tokenize import wordpunct_tokenize
import nltk
from nltk.stem.lancaster import LancasterStemmer
from encoder import label_encoder
import pickle
import json
import tflearn
import tensorflow as tf
import random

# Load components
nlp = spacy.load("en_core_web_md")
df = pd.read_csv("Database/Symptom-severity.csv")
df['Symptom'] = df['Symptom'].str.replace(' ','')
df['Symptom'] = df['Symptom'].str.replace('_',' ')

data = pickle.load(open("training_data","rb"))
database = pickle.load(open("data.pkl","rb"))

words = data["words"]
classes = data["classes"]
train_x = data["train_x"]
train_y = data["train_y"]
codes = database['codes']


with open("intents.json") as json_data:
    intents = json.load(json_data)

# Loading Chatbot model
net = tflearn.input_data(shape = [None, len(train_x[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(train_y[0]),activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net,tensorboard_dir='tflearn_logs')
model.load('model.tflearn')

# Loading stemmer
stemmer = LancasterStemmer()


def autocorrect(text):
    '''
    autocorrect(text)

    Performs basic spelling corrections in the language='en'.

    Parameter(s):
    ------------

        text : str
            Raw input as entered by the user prior to any form of preprocessing. 


    Returns:
    --------

        corrected : str
            A string with all words corrected to the best of the model's knowledge. All punctuation marks (except ',') are removed.
    '''
    
    spell = SpellChecker()
    
    low = wordpunct_tokenize(text)
    corrected = ""
    punctuations = """!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"""
    
    for word in low:
        if word not in punctuations:
            if (corrected != "") and (corrected[-1] != "'"):
                corrected += " " + spell.correction(word)
            else:
                corrected += spell.correction(word)
        else:
            corrected += word
    return corrected



def hasSymptoms(text, thresh=0.55):
    '''
    hasSymptom(text, thresh=0.5)

    Determines whether the sentence provided contains symptom-like words.
    
    Parameter(s):
    ------------

        text : str
            Spell-checked text entered by the user.

        thresh : float, optional, default: 0.5
            Minimum similarity threshold. If the similarity index between the text and symptoms database falls below thresh, 'False' is returned.


    Returns:
    --------

        possibility : bool
            Truth value of the text containing symptoms based on given threshold.
    '''

    low = text.replace(',',' ').split()

    possibility = 0
    for word in low:
        similarities = {}
        for symptom in df['Symptom'].values:
            similarities[symptom] = nlp(word).similarity(nlp(symptom))
        possibility = max(max(similarities.values()),possibility)
    return possibility > thresh

def strip(text):
    '''
    strip(text)
    
    Text preprocessor function. Removes stopwords and punctuations from text entered.

    Parameter(s):
    ------------

        text : str
            Spell-checked text entered by the user.


    Returns:
    --------

        filtered_sentence : str
            Processed text without stopwords and punctuations in the lowercase.
    '''

    stop_words = ['a',
                     'about',
                     'above',
                     'after',
                     'again',
                     'against',
                     'ain',
                     'all',
                     'am',
                     'an',
                     'any',
                     'are',
                     'aren',
                     "aren't",
                     'as',
                     'be',
                     'because',
                     'been',
                     'before',
                     'being',
                     'below',
                     'between',
                     'both',
                     'but',
                     'by',
                     'can',
                     'couldn',
                     "couldn't",
                     'd',
                     'did',
                     'didn',
                     "didn't",
                     'do',
                     'does',
                     'doesn',
                     "doesn't",
                     'doing',
                     'don',
                     "don't",
                     'down',
                     'during',
                     'each',
                     'few',
                     'for',
                     'from',
                     'further',
                     'had',
                     'hadn',
                     "hadn't",
                     'has',
                     'hasn',
                     "hasn't",
                     'have',
                     'haven',
                     "haven't",
                     'having',
                     'he',
                     'her',
                     'here',
                     'hers',
                     'herself',
                     'him',
                     'himself',
                     'his',
                     'how',
                     'i',
                     'if',
                     'into',
                     'is',
                     'isn',
                     "isn't",
                     'it',
                     "it's",
                     'its',
                     'itself',
                     'just',
                     'll',
                     'm',
                     'ma',
                     'me',
                     'mightn',
                     "mightn't",
                     'more',
                     'most',
                     'mustn',
                     "mustn't",
                     'my',
                     'myself',
                     'needn',
                     "needn't",
                     'no',
                     'nor',
                     'not',
                     'now',
                     'o',
                     'of',
                     'off',
                     'once',
                     'only',
                     'or',
                     'other',
                     'our',
                     'ours',
                     'ourselves',
                     'out',
                     'over',
                     'own',
                     're',
                     's',
                     'same',
                     'shan',
                     "shan't",
                     'she',
                     "she's",
                     'should',
                     "should've",
                     'shouldn',
                     "shouldn't",
                     'so',
                     'some',
                     'such',
                     't',
                     'than',
                     'that',
                     "that'll",
                     'the',
                     'their',
                     'theirs',
                     'them',
                     'themselves',
                     'then',
                     'there',
                     'these',
                     'they',
                     'this',
                     'those',
                     'through',
                     'to',
                     'too',
                     'under',
                     'until',
                     'up',
                     've',
                     'very',
                     'was',
                     'wasn',
                     "wasn't",
                     'we',
                     'were',
                     'weren',
                     "weren't",
                     'what',
                     'when',
                     'where',
                     'which',
                     'while',
                     'who',
                     'whom',
                     'why',
                     'will',
                     'with',
                     'won',
                     "won't",
                     'wouldn',
                     "wouldn't",
                     'y',
                     'you',
                     "you'd",
                     "you'll",
                     "you're",
                     "you've",
                     'your',
                     'yours',
                     'yourself',
                     'yourselves']
    
    punctuations = """!"#$%&'()*+ -./:;<=>?@[\]^_`{|}~"""
    

    word_tokens = wordpunct_tokenize(text)

    filtered_sentence = [w for w in word_tokens if (not w.lower() in stop_words) and (w not in punctuations)]
    
    return " ".join(filtered_sentence)


def symptomize(text):
    '''
    symptomize(text)

    Maps input from the user to the correctly worded symptoms that match the database.

    Parameter(s):
    ------------

        text : str
            Text entered by user after preprocessing.


    Returns:
    --------

        possibility : list
            Properly worded list of symptoms (corresponding to the symptoms database)
    '''

    los = text.replace(' and ',' , ').split(',')
    
    formatted = []
    
    for each in los:
        if each[0] == " ":
            each = each[1:]
        if each[-1] == " ":
            each = each[:-1]
        similarities = {}
        for symptom in df['Symptom'].values:
            similarities[symptom] = nlp(each).similarity(nlp(symptom))
        keymax = "Null" if max(similarities.values()) == 0 else max(similarities, key= lambda x: similarities[x])
        if keymax != "Null":
            formatted.append(keymax.replace(' ','_'))
    
    return formatted

def clean_up_sentence(sentence):
    '''
    Helper function to lemmatize given sentence
    '''
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    
    return sentence_words

def bow(sentence,words,show_details=False):
    '''
    Helper function to generate bag of words for entered text
    '''

    sentence_words = clean_up_sentence(sentence)
    
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag : %s" % w)
                    
    return(np.array(bag))


def classify(sentence,thresh=0.80):
    '''
    Helper function which runs the chatbot Neural Network to make prediction
    '''

    results = model.predict([bow(sentence,words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>thresh]
    # sort by strength of probability
    results.sort(key=lambda x: x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]],r[1]))
        
    return return_list

def response(sentence,show_details=False):
    '''
    Function that generates a response to user queries
    '''

    results = classify(sentence)
    
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    return random.choice(i['responses'])
                
            results.pop(0)
            
    else:
        for i in intents['intents']:
            if i['tag'] == "error":
                return i['responses'][0]

def pipeline(text,deja):
    '''
    Main pipeline function that interprets and evaluates text entered by the user.
    '''

    # Convert to lower case
    text = text.lower()
    
    # Correct spelling errors
    text = autocorrect(text)
    
    if text == "no":
        return deja

    output = response(text)
    
    if output.split()[0] == 'Sorry,':
    
        if hasSymptoms(text):

            # Preprocessing text
            text = strip(text)

            # Symptom mapper
            ls = symptomize(text)

            # Encoder
            array_of_symptoms = label_encoder(ls)
            if len(deja) != 0:
                array_of_symptoms.extend(deja)
                array_of_symptoms.sort(key=order)
                array_of_symptoms = array_of_symptoms[:17]

            return array_of_symptoms
    
    return(output)

def order(symptom_code):
    if symptom_code == 0:
        return 8
    symptom_name = list(codes.keys())[list(codes.values()).index(symptom_code)]
    symptom_name = symptom_name.replace(' ','').replace('_',' ')
    weight = int(df[df['Symptom'] == symptom_name]['weight'])
    return weight