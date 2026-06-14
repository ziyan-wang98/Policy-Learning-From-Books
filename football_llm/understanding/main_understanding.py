import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PLFB_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import ttt_simulator_info 
from understanding.book_reader import TicTacToeBooks2Knowledge
from CONFIG import *

if __name__ == "__main__":
    ttt_data_path = Path(os.environ.get("PLFB_TTT_DATA_PATH", Path(DATASET_PATH) / "tic_tac_toe_data"))
    book_path = Path(
        os.environ.get(
            "PLFB_TTT_BOOK_PATH",
            ttt_data_path / "tic_tac_toe_noise-knowledge-v2.jsonl",
        )
    )
    output_root = Path(
        os.environ.get(
            "PLFB_UNDERSTANDING_OUTPUT_ROOT",
            ttt_data_path / "book_knowledge",
        )
    )
    if not book_path.is_file():
        raise FileNotFoundError(
            f"Tic-Tac-Toe book file not found: {book_path}. "
            "Set PLFB_TTT_BOOK_PATH or PLFB_TTT_DATA_PATH."
        )
    print(f"PLFB_TTT_BOOK_PATH={book_path}")
    print(f"PLFB_UNDERSTANDING_OUTPUT_ROOT={output_root}")
    
    agg_model_name = 'gpt-4o'
    model_name = 'gpt-4o-mini'
    for knowledge_type in [KnowledgeType.Policy, KnowledgeType.Dynamics, KnowledgeType.Reward]:
        book_res_path = output_root / str(knowledge_type)
        os.makedirs(book_res_path, exist_ok=True)
        if knowledge_type == KnowledgeType.Policy:
            skip_rate = 0.0
        else:
            skip_rate = 0.8
        book_reader = TicTacToeBooks2Knowledge(skip_rate, str(book_path), str(book_res_path), [None],
                    model_name, agg_model_name, knowledge_type, 
                    ttt_simulator_info.TASK_DESC, ttt_simulator_info.element_type_prompt_dict, ttt_simulator_info.element_type_example_dict)
        book_reader.main_process()
