import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
from os.path import join as pjoin


class HFEntail:
  def __init__(self, entailment_model='facebook/bart-large-mnli', device=None) -> None:
    """_summary_

    Args:
        model (str, optional): _description_. Defaults to 'facebook/bart-large-mnli'.
    """
    self.nli = AutoModelForSequenceClassification.from_pretrained(entailment_model)
    self.tokenizer = AutoTokenizer.from_pretrained(entailment_model, use_fast=True)
    if device is None:
      self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

  def entail(self, premise, h):
    """_summary_

    Args:
        premise (_type_): _description_
        h (_type_): _description_

    Returns:
        _type_: _description_
    """
    hypothesis = [h for i in range(len(premise))]
    # tokenize the text
    try:
      x = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True).to(self.device)
      with torch.no_grad():
        nli_model = nli_model.to(self.device)
        logits = self.nli(x['input_ids'])[0]
        # we throw away "neutral" (dim 1) and take the probability of
        # "entailment" (2) as the probability of the label being true 
        # 0: contradicts, 1: neutral, 2: entailment
        entail_contradiction_logits = logits[:,[0,2]]
        # softmax probability of entailment
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:,1]
        batch_result = prob_label_is_true.tolist()
        return {'entail_ratio': batch_result}
    except ValueError:
      batch_result = [None for x in hypothesis]
      print("Some sort of error. Be Careful")
      x['entail_ration'] = batch_result
      pass
    return x