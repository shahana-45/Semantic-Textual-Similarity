import re
from torchtext.legacy.data import Field

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
        ## TODO normalize text to lower case       
        data['sentence_A'] = data['sentence_A'].apply(lambda x: x.lower())
        data['sentence_B'] = data['sentence_B'].apply(lambda x: x.lower())

        ## TODO remove punctuations
        data['sentence_A'] = data['sentence_A'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        data['sentence_B'] = data['sentence_B'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

        ## TODO remove stopwords
        
        ## TODO add any other preprocessing method (if necessary)
        return data
