from pipeline import Pipeline as Modelo_Definitivo
import constants
# Opciones de modelos
'''
log-reg
neural-net
rand-forest
grad-boosting
'''
tweet = ''
actual_model = 'neural-net'
while True:
    print('-'*50)
    print('Usign Model --> ', actual_model)
    print("Possible options --> ['exit', 'metrics', change-model]")
    tweet = input("Insert any tweet or option: ")
    if tweet.upper() == "SALIR" or tweet.upper() == "EXIT": break
    elif tweet.upper() == "METRICS": Modelo_Definitivo.get_metrics()
    elif tweet.upper() == "CHANGE-MODEL":
        newModel = input(f'Set the new Model {constants.LIST_MODELS}: ')
        if newModel in constants.LIST_MODELS:
            actual_model = newModel
            print('Model changed to', actual_model)
        else: print('Invalid Model', constants.LIST_MODELS)
        print('-'*50)
    else: Modelo_Definitivo.get_result(tweet, model=actual_model)




'''
lista = [
    'blame 100 islamist terrorist organizations book mormon',
    'hate girls',
    'fuck black people',
    'te odio',
    'otro tweet',
    'Los catolicos son una mierda, peor que los judios',
    'Catholics suck, worse than Jews',
    'you are a fucking fat guy, every one hates you nigga',
    'idiots rakshabandhan hindu festival hindus natur protect cows gut put poster tweet ani muslim festival shut get lost'
    ]
[Modelo_Definitivo.get_result(tweet) for tweet in lista]
'''
