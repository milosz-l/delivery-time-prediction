import pandas as pd


def convert_jsonl_to_csv(file_name, path=""):
    df = pd.read_json(path + file_name, lines=True)
    df.to_csv(f"data_in_csv/{file_name}.csv", index=False)


if __name__ == "__main__":
    convert_jsonl_to_csv("deliveries.jsonl", "data/")
    convert_jsonl_to_csv("products.jsonl", "data/")
    convert_jsonl_to_csv("sessions.jsonl", "data/")
    convert_jsonl_to_csv("users.jsonl", "data/")
