# WORD2ID_PKL = 'output/word2id.pkl'
# MODEL_PATH  = 'output/absa_bi-lstm.h5'

ASPECTS = ['SCREEN', 'CAMERA', 'FEATURES', 'BATTERY', 'PERFORMANCE', 
           'STORAGE', 'DESIGN', 'PRICE', 'GENERAL', 'SER&ACC']
REPLACEMENTS = {0: None, 1: 'Negative', 2: 'Neutral', 3: 'Positive'}

MAX_SEQUENCE_LENGTH = 256

PRETRAINED_MODEL = 'vinai/phobert-base'