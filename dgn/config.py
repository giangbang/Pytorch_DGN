import json


def read_dict_from_json(file_path):
    """Reads a dictionary from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The dictionary read from the JSON file.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def write_dict_to_json(data, file_path):
    """Writes a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to write.
        file_path (str): The path to the JSON file.
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
