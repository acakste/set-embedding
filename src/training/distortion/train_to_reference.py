

import torch


#************************************
def sample_one(reference_set):
    id = torch.randint(low=0, high=len(reference_set), size=(1,))
    return reference_set[id].unsqueeze(0), id


sample_fn_mappings = {
    "sample_one": sample_one
}

def select_sample_function(fn_str):
    try:
        return sample_fn_mappings[fn_str]
    except KeyError:
        print(f"{fn_str} is an invalid function.")


#************************************
