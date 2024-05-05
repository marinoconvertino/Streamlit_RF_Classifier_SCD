import os
import pickle

def load_models(file_path, numbers):
    models_dict = {}
    for r in numbers:
        file_name = f'RF_classifier_Fcell_r{r}.pkl'
        full_path = os.path.join(file_path, file_name)
        if os.path.exists(full_path):
            with open(full_path, 'rb') as file:
                key_name = f'model_r{r}'
                models_dict[key_name] = pickle.load(file)
        else:
            print(f'File {file_name} does not exist')
    return models_dict