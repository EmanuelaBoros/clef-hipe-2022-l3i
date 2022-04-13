__all__ = [
    "CamembertEmbedding",
    "BertEmbedding",
    "FlairEmbedding",
    "XLMRobertaEmbedding",
    "RobertaEmbedding"
]
from .camembert_embedding import CamembertEmbedding
from .bert_embedding import BertEmbedding
#from .flair_embedding import FlairEmbedding
from .roberta_embedding import RobertaEmbedding	
from .transformers_embedding import TransformersEmbedding	
from .xlm_roberta_embedding import XLMRobertaEmbedding	
