import os
import pickle
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo
import ssl
from deepforest_eoh.utils import get_dir_in_root


ssl._create_default_https_context = ssl._create_unverified_context


DATASET_CONFIG = {
    "Adult": {"source": "openml", "id": 179},
    "Arrhythmia": {"source": "openml", "id": 5},
    "BankMarketing": {"source": "ucirepo", "id": 222},
    "Car": {"source": "ucirepo", "id": 19},
    # "Covertype": {"source": "ucirepo", "id": 31},
    "CredictCard": {"source": "ucirepo", "id": 350},
    "Diabetes": {"source": "ucirepo", "id": 891},
    "DryBean": {"source": "ucirepo", "id": 602},
    "Gamma": {"source": "ucirepo", "id": 159},
    "HeartDisease": {"source": "ucirepo", "id": 45},
    "Maternal": {"source": "ucirepo", "id": 863},
    "Mushroom": {"source": "ucirepo", "id": 73},
    "Rice": {"source": "ucirepo", "id": 545},
    "Student": {"source": "ucirepo", "id": 697},
    "Websites":{"source": "ucirepo", "id": 327},
    "Letter":{"source": "ucirepo", "id": 59},
    "HTRU2":{"source": "ucirepo", "id": 372},

}

CACHE_DIR = get_dir_in_root("dataset_raw")
os.makedirs(CACHE_DIR, exist_ok=True)


def download_and_save(dataset_name, dataset_dict):
    source = dataset_dict["source"]
    data_id = dataset_dict["id"]

    if source == "openml":
        dataset = fetch_openml(data_id=data_id)
        data = dataset
    elif source == "ucirepo":
        dataset = fetch_ucirepo(id=data_id)
        data = {"features": dataset.data.features, "targets": dataset.data.targets}
    else:
        raise ValueError(f"Unknown source '{source}' for dataset '{dataset_name}'")

    # save as .pkl
    save_path = os.path.join(CACHE_DIR, f"{dataset_name}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved dataset '{dataset_name}' to {save_path}")


def main():
    for name, dataset_dict in DATASET_CONFIG.items():
        download_and_save(name, dataset_dict)


if __name__ == "__main__":
    main()
