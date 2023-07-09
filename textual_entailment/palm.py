import google.generativeai as palm
import time

class PALM:
    def __init__(self, key, model_idx=0, max_retry=5, json=True) -> None:
      palm.configure(api_key=key)
      self.models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
      self.model = self.models[model_idx].name
      self.max_retry = max_retry
      self.json = json

    def prompt(self, premise, hypothesis):
      addendum = ''
      if self.json:
        addendum += 'Return your answer only in JSON.'
      return f"""
        You are a natural language inference model. Your task is to see if the given hypothesis entails the given premise. You must also quantify as a value between 0 and 1 the probability of entailment between the premise and hypothesis. You must also explain why you chose this value for the probability of entailment. {addendum}
        Premise:
        {premise}
        Hypothesis:
        {hypothesis}
      """, 2048
    
    def entail(self, x):
      text = f"{x['title']}\n{x['post']}\n{x['comment']}"
      text, max_tokens = self.prompt(text)
      for i in range(self.max_retry):
        try:
          answer = palm.generate_text(
            model=self.model,
            prompt=f"{text}",
            temperature=0,
            safety_settings=[
              {
                'category':c, 
                'threshold': palm.types.HarmBlockThreshold.BLOCK_NONE
              } for c in palm.types.HarmCategory
            ],
            max_output_tokens=max_tokens
          )
          x['candidates'] = answer.candidates
          return x
        except Exception as e:
          print(e)
          time.sleep(60)