import pickle
import sklearn
import warnings

from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter("error", InconsistentVersionWarning)

try:
    with open('saved_model.sav', 'rb') as file:
        model = pickle.load(file)
except InconsistentVersionWarning as w:
    print(f"Scikit-learn version mismatch. Original version: {w.original_sklearn_version}")
except FileNotFoundError:
    print("Error: 'saved_model.sav' file not found.")
except EOFError:
    print("Error: End of file while unpickling the model.")
except pickle.UnpicklingError as e:
    print(f"Error loading the model: {e}")
except ModuleNotFoundError:
    print("Error: scikit-learn module not found.")
except Exception as e:
    print(f"Unexpected error: {e}")
