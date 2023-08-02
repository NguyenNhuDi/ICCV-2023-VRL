# importing the module
import json
import argparse

 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    prog='Model Trainer',
    description='This program will train a model',
    epilog='Vision Research Lab')
    parser.add_argument('-c', '--config', required=True,
                        help='The path to the config file.')
    args = parser.parse_args()


    with open(args.config) as f:
        args = json.load(f)

    json_paths = args['json_paths']
    file_path = args['save_path']

    all_json = []

    json_dict = {
        "test_dir": "",
        "save_path": "",
        "batch_size": "",
        "all_month_sizes": [],
        "all_month_means": [],
        "all_month_stds": [],
        "march_sizes": [],
        "april_sizes": [],
        "may_sizes": [],
        "march_means": [],
        "march_stds": [],
        "april_means": [],
        "april_stds": [],
        "may_means": [],
        "may_stds": [],
        "march_models": [],
        "april_models": [],
        "may_models": [],
        "run_amount": 1
    }

    for j in json_paths:
        all_json.append(json.load(open(j)))


    for j in all_json:
        for key in j:
            curr_item = j[key]

            if type(curr_item) == type([]):
                for i in curr_item:
                    json_dict[key].append(i)


    json_string = json.dumps(json_dict, indent=4)
    with open(file_path, "w") as json_file:
        json_file.write(json_string)
