from googleapiclient import discovery
import constants
import os

class Identifier:
    def __init__(self) -> None:
        self.client = discovery.build(
          "commentanalyzer",
          "v1alpha1",
          developerKey=os.environ["API_KEY_GOOGLE"],
          discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
          static_discovery=False,
        )
    
    def get_toxicity(self, tweet):
      analyze_request = {'comment': { 'text':  tweet},'requestedAttributes': {'TOXICITY': {}}}
      response = self.client.comments().analyze(body=analyze_request).execute()
      score = response["attributeScores"]["TOXICITY"]["spanScores"][0]["score"]["value"]
      return {'score':score, 'isBullying':bool(score > constants.TOXIC_THRESHOLD), 'language':response["languages"]}