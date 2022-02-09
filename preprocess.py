import re
from torchtext.legacy.data import Field
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords

"""
Performs basic text cleansing on the unstructured field 
"""


class Preprocess:
    def __init__(self, stpwds_file_path):
        """
        Initializes regex patterns and loads stopwords
        """
        # TODO implement

    def perform_preprocessing(self, data, columns_mapping):
        ## normalize text to lower case       
        data['sentence_A'] = data['sentence_A'].apply(lambda x: x.lower())
        data['sentence_B'] = data['sentence_B'].apply(lambda x: x.lower())

        ## remove punctuations
        data['sentence_A'] = data['sentence_A'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        data['sentence_B'] = data['sentence_B'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

        ## remove stopwords
        data['sentence_A'] = data['sentence_A'].apply(lambda x: remove_stopwords(x))
        data['sentence_B'] = data['sentence_B'].apply(lambda x: remove_stopwords(x))
        
        ## TODO add any other preprocessing method (if necessary)
        return data
