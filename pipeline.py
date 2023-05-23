from cleaningDataset import Cleaner
from identifyBullying import Identifier
from classifyBullying import Classifier
import constants

# Limpia data
# dataCleaner = Cleaner()
# dataCleaner.cleanDataset()

# Initialize toxicity identifier
bullyIdentifier = Identifier()

# Initialize cyberbullying classifier
bullyClassifier = Classifier()

class Pipeline:

    # pipeline for a tweet, input a string twwet, then tweet is cleaned, parsed and passed to the model, finally it is identified and classfied
    def get_result(tweet, model='neural-net'):
        print('-'*50)
        identify = bullyIdentifier.get_toxicity(tweet)
        print('Tweet:', tweet)
        print("Identify:", identify)
        if identify["isBullying"]:
            response = bullyClassifier.load_predict(model, [tweet])
        else:
            response = "No es ofensivo"
        print("Classify:", response)
        print('-'*50)
        return response

    # Get metrics of models
    def get_metrics():
        for model in constants.LIST_MODELS:
            print('-'*50, model,'-'*50)
            bullyClassifier.load_predict(model)
        print('-'*50)

    # Train models
    def train_models():
        for model in constants.LIST_MODELS:
            bullyClassifier.trainModel(model)