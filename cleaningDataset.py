
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.experimental import enable_halving_search_cv
import pickle
import constants

# Stop words are a set of commonly used words in a language.
stp_words=set(stopwords.words('english'))
stp_words.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 
                  'im', 'll', 'y', 've', 'u', 'ur', 'don', 
                  'p', 't', 's', 'aren', 'kp', 'o', 'kat', 
                  'de', 're', 'amp', 'will', 'wa', 'e', 'like'])

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }
# spell=Speller(lang='en')
stemmer=SnowballStemmer('english')

class Cleaner:
    def clean_text(self, text):
        #removing hastags and links
        pattern=re.compile(r"(#[a-zA-Z0-9]+|@[a-zA-Z0-9]+|https?://\S+|www\.\S+|\S+\.[a-z]+|RT @)")
        text=pattern.sub('',text)
        text=" ".join(text.split())
        
        #make all lowercase
        text=text.lower()
        
        #stemming
        text=" ".join([stemmer.stem(word) for word in text.split()])
        
        #remove shortands
        s=''
        for word in text.split():
            if word in CONTRACTION_MAP.keys():
                s=s+' '+CONTRACTION_MAP[word];
            else:
                s=s+' '+word
        text=s
        
        #remove puncutations
        punc=re.compile(r"[^\w\s]")
        text=punc.sub('',text)
        
        #remove stop words
        text=" ".join(word for word in text.split() if word not in stp_words)
        
        #remove emojis
    #     emoji=demoji.findall(text)
    #     for emoj in emoji:
    #         text = re.sub(r"(%s)" % (emot), "_".join(emoji[emot].split()), text)
            
        #applying autocorrect to words
    #     correct_spell=''
    #     for word in text.split():
    #         correct_spell=correct_spell+' '+spell(word)
    #     text=" ".join(correct_spell.split())
        return text

    def cleanDataset(self):
        # Se lee el dataset
        df=pd.read_csv(constants.CSV_INPUT)

        # print los primeros 5 elementps
        # print(df.head())
        # Conteo de las clases Y
        # print(df['cyberbullying_type'].value_counts())

        # import seaborn as sns
        # sns.countplot(data=df,x='cyberbullying_type')

        # clean each tweet
        df['clean_data']=df['tweet_text'].apply(lambda tweet: self.clean_text(tweet))
        del df["tweet_text"]

        # print sampleo de 10 tweets que ya estan limpios
        # print(df.sample(10))

        # imprime los tweets duplicados
        # print(df.clean_data.duplicated().sum())

        # Elimina los tweets duplicados
        df.drop_duplicates('clean_data', inplace=True)

        # check if there is only spaces in cleaned data, tambien revisa los strings vacios
        # print(df['clean_data'].str.isspace().sum())
        df['clean_data'].replace('', np.nan, inplace=True)
        df.dropna(inplace=True)

        # Print conteo de los elementos nulos
        # print(df.isna().sum())

        # Print conteo de los elementos nulos
        # print(df.isnull().sum())

        # removing unidentified cyberbullying type
        df=df[df['cyberbullying_type']!='other_cyberbullying']

        # print(df.sample(5))

        # Encoding the cyberbullying_type column, pasa cada clase string a un mapeo de clase 0,1,2 etc
        from sklearn.preprocessing import LabelEncoder
        encoder=LabelEncoder()
        df.cyberbullying_type=encoder.fit_transform(df['cyberbullying_type'])

        # print data con columna y tranformada, e informacion de los parametros del encoder
        # print(df)
        # print(encoder.get_params())

        # save labelEncoder
        pickle.dump(encoder, open(f"{constants.MODELS_PATH}{constants.LBL_ENCODER_FILE}", 'wb'))

        # saving the dataframe
        df.to_csv(constants.CSV_CLEANED, header=True, index=False, encoding='utf-8')









