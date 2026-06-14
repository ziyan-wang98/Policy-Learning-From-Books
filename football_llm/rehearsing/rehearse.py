
from retrieval.state_filed_retrieval import KnowledgeRetrieval

class CodeInstantiation(object):
    def __init__(self, retrieval_model:KnowledgeRetrieval, cache_path:str, text_to_obs_fn:callable):
        self.retrieval_model = retrieval_model
        self.cache_path = cache_path
        self.text_to_obs_fn = text_to_obs_fn
    
    def get_code(self, text_obs:str):
        return self.retrieval_model.get_code(text_obs)
        