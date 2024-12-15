import pandas as pd
import json

if __name__ == "__main__":
    file_path = "lego.xlsx" 
    data = pd.read_excel(file_path)

    name_to_class = {row["Name"]: int(row["class"]) for _, row in data.iterrows()}
    class_to_name = {int(row["class"]): row["Name"] for _, row in data.iterrows()}

    with open("name_to_class.json", "w") as name_to_class_file:
        json.dump(name_to_class, name_to_class_file, indent=4)

    with open("class_to_name.json", "w") as class_to_name_file:
        json.dump(class_to_name, class_to_name_file, indent=4)

    print("JSON-files created successfully: 'name_to_class.json' and 'class_to_name.json'")
