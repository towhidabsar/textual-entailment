import json
from allennlp_models import pretrained
import datasets
import numpy as np
from os.path import join as pjoin

class AllenNLPEntail:
  """_summary_
  """
  def __init__(self, entailment_model, cuda_device=-1) -> None:
    """_summary_

    Args:
        entailment_model (str): The name of the Allen NLP model to initialize.
        cuda_device (int, optional): Whether to run on CUDA since it's available. Defaults to -1 for CPU.
    """
    self.nli = pretrained.load_predictor(
        entailment_model, 
        cuda_device=cuda_device
    )
    self.labels = ['entailment', 'contradict', 'neutral']

  def entail(self, premise, hypothesis):
    """
    _summary_
    Args:
        premise (str): The premise of the textual entailment
        hypothesis (str): The hypothesis of the textual entailment

    Returns:
        (dict): Returns the type of entailment and the ratio of entailment  
    """
    p = self.nli.predict(premise=premise, hypothesis=hypothesis)
    entail_idx = np.argmax(p['probs'])
    entail_ratio = p['probs'][entail_idx]
    verdict = self.labels[entail_idx]
    return { 'verdict': verdict, 'ratio': entail_ratio }