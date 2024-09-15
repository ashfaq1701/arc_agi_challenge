import json

import numpy as np


def read_training_data():
    with open('data/arc-agi_training_challenges.json', 'r') as file:
        training_data = json.load(file)

    with open('data/arc-agi_training_solutions.json', 'r') as file:
        training_solutions = json.load(file)

    combined_training_data = {}

    for key, group in training_data.items():
        test_solutions = training_solutions[key]
        training_pairs = group['train']
        testing_data = group['test']

        for i in range(len(testing_data)):
            testing_data[i]['output'] = test_solutions[i]

        combined_training_data[key] = {
            'train': training_pairs,
            'test': testing_data
        }

    return convert_samples_to_numpy(combined_training_data)


def read_evaluation_data():
    with open('data/arc-agi_evaluation_challenges.json', 'r') as file:
        evaluation_data = json.load(file)

    with open('data/arc-agi_evaluation_solutions.json', 'r') as file:
        evaluation_solutions = json.load(file)

    combined_evaluation_data = {}
    for key, group in evaluation_data.items():
        test_solutions = evaluation_solutions[key]
        training_pairs = group['train']
        testing_data = group['test']

        for i in range(len(testing_data)):
            testing_data[i]['output'] = test_solutions[i]

        combined_evaluation_data[key] = {
            'train': training_pairs,
            'test': testing_data
        }

    return convert_samples_to_numpy(combined_evaluation_data)


def read_test_data():
    with open('data/arc-agi_test_challenges.json', 'r') as file:
        test_data = json.load(file)

    return convert_samples_to_numpy(test_data)


def convert_samples_to_numpy(data_samples):
    data_samples_np = {}

    for key, group in data_samples.items():
        train = group['train']
        test = group['test']

        train_np = []
        test_np = []

        for train_item in train:
            img_pair = {
                'input': np.array(train_item['input']),
                'output': np.array(train_item['output'])
            }
            train_np.append(img_pair)

        for test_item in test:
            img_pair = {'input': np.array(test_item['input'])}
            if 'output' in test_item:
                img_pair['output'] = np.array(test_item['output'])
            test_np.append(img_pair)

        data_samples_np[key] = {'train': train_np, 'test': test_np}

    return data_samples_np


def get_max_dims(datasets):
    max_height = 0
    max_width = 0

    for dataset in datasets:
        for key, group in dataset.items():
            train_data = group['train']
            for img_pair in train_data:
                input_img = img_pair['input']
                output_img = img_pair['output']

                input_height, input_width = input_img.shape
                output_height, output_width = output_img.shape

                max_height = max(max_height, input_height, output_height)
                max_width = max(max_width, input_width, output_width)

            test_data = group['test']
            for img_pair in test_data:
                input_img = img_pair['input']
                input_height, input_width = input_img.shape
                max_height = max(max_height, input_height)
                max_width = max(max_width, input_width)

                if 'output' in img_pair:
                    output_img = img_pair['output']
                    output_height, output_width = output_img.shape
                    max_height = max(max_height, output_height)
                    max_width = max(max_width, output_width)

    return max_height, max_width
