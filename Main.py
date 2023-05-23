from pipeline import Pipeline as Modelo_Definitivo
# Opciones de modelos
'''
log-reg
neural-net
rand-forest
grad-boosting
'''
tweet = ''
while True:
    tweet = input("Insert any tweet: ")
    if tweet.upper() == "SALIR" or tweet.upper() == "EXIT": break
    Modelo_Definitivo.get_result(tweet)




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
