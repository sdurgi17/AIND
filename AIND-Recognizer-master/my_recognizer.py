import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    test_data = test_set.get_all_Xlengths()
    for idx in test_data:
        word_probability_dict = {}
        best_guess_logL = float("-inf")
        best_guess = None
        for word in models:
            # print(test_data[idx][0], test_data[idx][1])
            try:
                word_probability_dict[word] = models[word].score(test_data[idx][0], test_data[idx][1])
            except:
                word_probability_dict[word] = float("-inf")
            if word_probability_dict[word] > best_guess_logL:
                best_guess_logL = word_probability_dict[word]
                best_guess = word
        guesses.append(best_guess)
        probabilities.append(word_probability_dict)
    return (probabilities, guesses)
                
            