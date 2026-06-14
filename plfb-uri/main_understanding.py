from omegaconf import DictConfig, OmegaConf
import hydra

from openai import OpenAI
import openai
import sys
from understanding.book_reader import TicTacToeBooks2Knowledge, FootballBooks2Knowledge
from const import *
import os


@hydra.main(version_base=None, config_path="configs/", config_name="conf")
def main(cfg: DictConfig) -> None:
    book_path = cfg.path.book_path
    agg_model_name = os.environ.get('PLFB_OPENAI_AGG_MODEL', 'gpt-4o')
    model_name = os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini')
    for knowledge_type in [KnowledgeType.Policy, KnowledgeType.Dynamics, KnowledgeType.Reward]:
        book_res_path = os.path.join(cfg.path.res_path, 'book_knowledge', knowledge_type)
        os.makedirs(book_res_path, exist_ok=True)
        if cfg.sim_info.ENV_NAME == 'tictactoe':
            if knowledge_type == KnowledgeType.Policy:
                skip_rate = 0.0
            else:
                skip_rate = 0.8
            
            book_reader = TicTacToeBooks2Knowledge(skip_rate, book_path, book_res_path, [None], 
                        model_name, agg_model_name, knowledge_type, 
                        cfg.sim_info.TASK_DESC, cfg.sim_info.element_type_prompt_dict, cfg.sim_info.element_type_example_dict, 
                        agg_res_prefix=cfg.path.agg_res_prefix)
        elif cfg.sim_info.ENV_NAME == 'football':
            book_reader = FootballBooks2Knowledge(book_path, book_res_path, [None], 
                        model_name, agg_model_name, knowledge_type, 
                        cfg.sim_info.TASK_DESC, cfg.sim_info.element_type_prompt_dict, cfg.sim_info.element_type_example_dict, 
                        agg_res_prefix=cfg.path.agg_res_prefix)
        else:
            raise NotImplementedError
            
        book_reader.main_process()
    
if __name__ == "__main__":
    main()
