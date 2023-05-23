from cleaningDataset import Cleaner
from identifyBullying import Identifier
from classifyBullying import Classifier

# Limpia data
# dataCleaner = Cleaner()
# dataCleaner.cleanDataset()

# Initialize toxicity identifier
bullyIdentifier = Identifier()

# Prepara o Entrena los modelos de clasificacion
bullyClassifier = Classifier()
bullyClassifier.prepare_train()
# bullyClassifier.trainModel('log-reg')
# bullyClassifier.trainModel('grad-boosting')
# bullyClassifier.trainModel('rand-forest')
# bullyClassifier.trainModel('neural-net')

# bullyClassifier.load_predict('grad-boosting')
# bullyClassifier.load_predict('log-reg')
# bullyClassifier.load_predict('rand-forest')

class Pipeline:
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