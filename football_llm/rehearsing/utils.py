import io
import os
import os.path as osp
from pathlib import Path
import numpy as np
from funcs import npz_extractor

PROJECT_ROOT = Path(os.environ.get("PLFB_ROOT", Path(__file__).resolve().parents[1])).resolve()


def get_sampled_data(dir_path, load_num=50):
    """
    Randomly pick a file from dir_path and load the data.
    Keep trying until a file with the key 'wrapper_obs' is found.
    """
    
    save_dir = osp.join(dir_path, 'sample_data')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, f'sample_{load_num}-v6.npz')
    npz_files = {}
    counter = 0
    # check if already exists a file named sample_{load_num}.npz
    if os.path.exists(save_path):
        offline_data = np.load(save_path, allow_pickle=True)
        if len(offline_data.keys()) < load_num:
            print("sample file is not enough, re-sample")
            counter = len(offline_data)
            npz_files = {k: v for k, v in offline_data.items()}
        else:
            return offline_data
    else:
        counter = 0
    for file in os.listdir(dir_path):
        if file.endswith('.npz'):
            res = npz_extractor(osp.join(dir_path, file))
            if res is None:
                continue
            if max(res['reward']) == 0:
                continue
            if file in npz_files:
                continue
            print("good trajectory", file)
            with open(osp.join(dir_path, file), 'rb') as f:
                npz_files[file] = f.read()
            counter += 1
            if counter >= load_num:
                break
    np.savez(save_path, **npz_files)
    # save the log
    offline_data = np.load(save_path, allow_pickle=True)
    return offline_data



def load_inner_npz_by_index(data, index):
    keys = list(data.keys())  # Convert the keys to a list
    if index < len(keys):
        inner_file_key = keys[index]  # Get the key at the specified index
        npz_binary = data[inner_file_key]
        npz_data = np.load(io.BytesIO(npz_binary), allow_pickle=True)
        return npz_data
    else:
        raise IndexError(f"Index {index} is out of bounds for the number of files.")


def start_point_picker_v2(offline_data, n, interval=10):
    start_point_list = []
    data = load_inner_npz_by_index(offline_data, n)
    for j in range(int(len(data['steps']) / interval)):
        for _ in range(interval * 2):
            num = np.random.randint(-interval//2, interval//2)
            start_point = j * interval + num
            if data['obs_before_modified_by_acs'][start_point][0]['game_mode'] == 0:
                start_point_list.append([n, start_point])
                break
    return start_point_list 



def load_bc_models(bc_root_path=None):
    if bc_root_path is None:
        bc_root_path = os.environ.get(
            "PLFB_BC_MODEL_ROOT",
            str(PROJECT_ROOT / "ORL_LOG_BC-v4" / "d3rlpy_logs"),
        )
    bc_models = {}
    
    import d3rlpy
    
    for i in range(22):
        file_start = f"comment=final-nz-10x-fix-feat&stack_hist=10&stack_obs=3&batch_size=4096&pl={i}&"
        # find the file start with file_start
        for file in os.listdir(bc_root_path):
            if file.startswith(file_start):
                exp_path = os.path.join(bc_root_path, file)
                train_episodes = 500000
                path = model_path =  os.path.join(exp_path, f'model_{train_episodes}.d3')
                player_id = path.split('pl=')[1].split('_')[0].split('&')[0]
                stack_hist = path.split('stack_hist=')[1].split('&')[0]
                stack_obs_num = path.split('stack_obs=')[1].split('&')[0]
                bc = d3rlpy.load_learnable(path)
                bc_models[i] = bc
    return bc_models, stack_hist, stack_obs_num
