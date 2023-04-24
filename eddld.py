import os
import pandas as pd
import yaml
import numpy as np
import shutil


dataset_root = "G:\dataset\eddld"
image_root = os.path.join(dataset_root, "images")
export_root = os.path.join(dataset_root, "export")
num_clients = 4
iid = False
alpha = 0.3
seed = 0
n_classes = 4


def set_seed(seed):
    np.random.seed(seed)


def partation_data(data, num_clients, iid, alpha):
    num_samples = len(data)

    if iid:
        num_samples_per_client = int(num_samples / num_clients)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        client_data = {
            i: data.iloc[
                indices[i * num_samples_per_client : (i + 1) * num_samples_per_client]
            ]
            for i in range(num_clients)
        }

    else:

        label_distribution = np.random.dirichlet([alpha] * num_clients, n_classes)

        class_idcs = [
            np.argwhere(np.array(data.category) == i).flatten()
            for i in range(n_classes)
        ]

        client_idcs = [[] for _ in range(num_clients)]

        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(
                np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
            ):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        client_data = {i: data.iloc[client_idcs[i]] for i in range(num_clients)}

    return client_data


def main():
    set_seed(seed)
    data = pd.read_csv(dataset_root + "/labels.csv")
    client_data = partation_data(data, num_clients, iid, alpha)

    export_root = os.path.join(
        dataset_root, "export", "iid" if iid else f"non-iid_{alpha}"
    )
    os.makedirs(export_root, exist_ok=True)

    for i in range(num_clients):
        client_export_root = os.path.join(export_root, f"client_{i}")
        os.makedirs(client_export_root, exist_ok=True)
        os.makedirs(os.path.join(client_export_root, "images"), exist_ok=True)

        # images, meta.yaml, train.csv

        # client_data columns: name, category, type, grade -> name, label
        client_data[i] = client_data[i].rename(columns={"category": "label"})

        client_data[i].drop(columns=["type", "grade"], inplace=True)
        client_data[i].label = client_data[i].label.astype(int)
        client_data[i].name = client_data[i].name.astype(str).str.replace(".jpg", "")

        client_data[i].to_csv(
            os.path.join(client_export_root, "train.csv"), index=False
        )

        with open(os.path.join(client_export_root, "meta.yaml"), "w") as f:
            yaml.dump(
                {
                    "attributes": {
                        "evalType": "multi",
                        "name": f"Eye Disease Deep Learning Dataset {i} of {num_clients}",
                        "note": f"This dataset is generated by EDDLD dataset with {iid if iid else f'non-iid alpha={alpha}'} partitioning.",
                    },
                    "inputs": {
                        "ext": "jpg",
                        "pilMode": "RGB",
                        "type": "image",
                    },
                    "labelMappping": [
                        {"label": "0", "mapped": "point-like corneal ulcers"},
                        {"label": "1", "mapped": "point-flaky mixed corneal ulcers"},
                        {"label": "2", "mapped": "flaky corneal ulcers"},
                    ],
                    "targets": [
                        {"name": "label", "type": "integer"},
                    ],
                    "type": "vision",
                },
                f,
            )

        for image_nm in client_data[i].name:
            image_nm = image_nm + ".jpg"
            shutil.copy(
                os.path.join(image_root, image_nm),
                os.path.join(client_export_root, "images", image_nm),
            )

        print(f"Client {i} has {len(client_data[i])} samples exported.")


if __name__ == "__main__":
    main()
