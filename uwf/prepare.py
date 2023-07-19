import os
import pandas as pd
import shutil

data_path = os.path.join("ultra-widefield_images", "ultra-widefield-training")
label_path = os.path.join(data_path, "ultra-widefield-training.csv")
# img_path = os.path.join(data_path, "Images")

class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

output_path = os.path.join(data_path, "output")

if __name__ == "__main__":
    df = pd.read_csv(label_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for idx, class_names in enumerate(class_names):
        class_path = os.path.join(output_path, class_names)
        if not os.path.exists(class_path):
            os.mkdir(class_path)

        df_class = df[df["DR_level"] == idx]

        for i, row in df_class.iterrows():
            img_path = row["image_path"]
            # remove first folder name
            img_path = os.path.join(*img_path.split("\\")[1:])
            img_path = os.path.join(data_path, "Images", img_path)
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(class_path, img_name))

        print("Finish class: ", class_names, " with ", len(df_class), " images")
