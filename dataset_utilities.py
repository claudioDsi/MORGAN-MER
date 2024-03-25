import os
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import networkx as nx
import re
import config as cf
import xml.etree.ElementTree as ET
import tkinter as tk
from jinja2 import Environment, FileSystemLoader
import shutil
import random


def preprocess_term(term):
    return term.split(',')[0].split(' ')[0].replace('(', '').replace(')', '')

def get_attributes_from_metaclass(metaclass):
    list_attrs= metaclass.split(' ')[1:-1]
    list_results=[]
    for attr in list_attrs:
        list_results.append(attr.split(',')[0].replace('(', ''))

    return list_results



def create_tuple_list(label_list, data_list):
    list_tuple=[]
    for l, d in zip(label_list, data_list):
        tuple_data = l, d
        list_tuple.append(tuple_data)

    return list_tuple

def split_dataset(filename):
    labels = []
    test_docs = []
    train_docs = []
    with open(filename, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            if line.find('\t') != -1:
                content = line.split('\t')
                labels.append(content[0])
                graph_tot = content[1].split(" ")[:-1]
                size = (len(graph_tot * 2) / 3)
                split_test = graph_tot[int(size): -1]
                string_train = ' '.join([str(elem) for elem in graph_tot])
                string_test = ' '.join([str(elem) for elem in split_test])
                train_docs.append(string_train)
                test_docs.append(string_test)

    return train_docs, test_docs, labels


def load_file(filename):
    labels = []
    docs = []
    with open(filename,'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            if line.find('\t')!=-1:
                content = line.split('\t')
                if len(content) > 0:
                    labels.append(content[0])
                    docs.append(content[1])
    return  docs


def get_gt_classes(filename):
    docs = []
    try:
        with open(filename,'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                if line.find('\t')!= -1:
                    content = line.split('\t')
                    #docs.append(content[1].split(' ')[0])
                    docs.append(tuple(content[1].split(' ')))
    except:
        print(filename)
        return None
    return docs



def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    ## nlp here ##
    #string = re.sub(r"\(", "", string)
    #string = re.sub(r"\)", "", string)
    ##
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def find_unique_values(train_data):
    with open("unique_values.txt", "w", encoding="utf8", errors="ignore") as res:
        for train in train_data:
            attrs = train.split(" ")
            unique = set(attrs)
            for u in unique:
                res.write(u + "\n")


def preprocessing(docs):
    preprocessed_docs = []
    stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()

    for doc in docs:
        clean_doc = clean_str(doc)
        new_values = []
        #print(new_values)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
        #preprocessed_docs.append([wordnet_lemmatizer.lemmatize(w) for w in clean_doc])
    return preprocessed_docs


def get_vocab_train(train_docs):
    vocab = dict()
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def get_vocab(train_docs, test_docs):
    vocab = dict()

    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def create_graphs_of_words(docs, vocab, window_size):
        graphs = list()
        for idx, doc in enumerate(docs):
            G = nx.Graph()
            for i in range(len(doc)):
                if doc[i] not in G.nodes():
                    G.add_node(doc[i])
                    G.nodes[doc[i]]['label'] = vocab[doc[i]]
            for i in range(len(doc)):
                for j in range(i + 1, i + window_size):
                    if j < len(doc):
                        G.add_edge(doc[i], doc[j])

            graphs.append(G)

        return graphs


def convert_string_to_list(list_element):
    str_format=''
    return str_format.join(list_element)


def encoding_data(train_context):

    data = load_file(train_context)
    train_preprocessed = preprocessing(data)
    find_unique_values(data)
    return train_preprocessed, data


def encoding_training_data_dump(train_context):
    train_data = load_file(train_context)
    train_preprocessed = preprocessing(train_data)
    find_unique_values(train_data)

    # Store the preprocessed data in a file using pickle
    with open(cf.DUMP_TRAIN, "wb") as f:
        pickle.dump(train_preprocessed, f)

    return train_preprocessed, train_data


def load_preprocessed_data(filename):
    with open(filename, "rb") as f:
        train_preprocessed = pickle.load(f)
    return train_preprocessed


def preprocess_test_data(test_context):
    y_test, test_data = load_file(test_context)
    test_preprocessed = preprocessing(test_data)
    return test_preprocessed


def match_operations(predicted, actual, gt_data):
    match = [value for value in predicted if value in actual]
    dict_op={}
    for op in gt_data:
        splitted_op = op.split(" ")
        for rec in match:
            if splitted_op[0] == rec:
                dict_op.update({rec: splitted_op})

    return dict_op


def success_rate(predicted, actual, n):
    if actual:
        match = [value for value in predicted if value in actual]

        if len(match) >= n:
            return 1
        else:
            return 0
    else:
        return 0



def precision(predicted,actual):
    if actual and predicted:
        true_p = len([value for value in predicted if value in actual])
        false_p = len([value for value in predicted if value not in actual])
        return (true_p / (true_p + false_p))*100
    else:
        return 0


def recall(predicted,actual):
    if actual and predicted:
        # true_p = len([value for value in predicted if value in actual])
        false_n = len([value for value in actual if value not in predicted])
        true_p = len([value for value in predicted if value in actual])
        return (true_p/(true_p + false_n))*100
    else:
        return 0


def format_dict(dict):

    out_string = ""
    i = 0
    for key, value in dict.items():
        out_string += str(key)+":"+str(value)+"#"
    return out_string


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_xes_traces(in_path, out_path, is_train):
    create_path_if_not_exists(out_path)
    if is_train:
        with open(f"{out_path}train.txt", 'w', encoding='utf8', errors='ignore') as res:
            for file in os.listdir(in_path):
                try:
                    tree = ET.parse(in_path + file)
                    root = tree.getroot()
                    for trace in root.findall('trace'):
                        if len(trace) > 0:
                            for event in trace:
                                event_data = {'class': '', 'featureName': '', 'eventType': ''}
                                for attribute in event:
                                    key = attribute.attrib.get('key')
                                    if key in event_data:
                                        event_data[key] = attribute.attrib.get('value', '')
                                res.write(
                                    f"event\t{event_data['class']} {event_data['featureName']} {event_data['eventType']}\n")

                except:
                    print("No log trace in file", file)
    else:
        for file_name in os.listdir(in_path):
            input_file_path = os.path.join(in_path, file_name)
            output_file_path = os.path.join(out_path, file_name)

            try:
                with open(input_file_path, 'r', encoding='utf8') as input_file:
                    tree = ET.parse(input_file)
                    root = tree.getroot()

                with open(output_file_path, 'w', encoding='utf8') as res:
                    for trace in root.findall('trace'):
                        if len(trace) > 0:
                            for event in trace:
                                event_data = {'class': '', 'featureName': '', 'eventType': ''}
                                for attribute in event:
                                    key = attribute.attrib.get('key')
                                    if key in event_data:
                                        event_data[key] = attribute.attrib.get('value', '')
                                res.write(
                                    f"event\t{event_data['class']} {event_data['featureName']} {event_data['eventType']}\n")

            except ET.ParseError:
                print(f"No log trace in file or XML parsing error: {file_name}")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")






def aggregate_cluster_files(path, outpath, filename):
    with open(outpath + filename, 'wb') as wfd:
        for f in os.listdir(path):
            try:
                with open(path + f, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
            except:
                continue


def creates_train_file(directory_path, output_file_path):
    """
    Reads the content of all files in the specified directory and writes it into a single output file.

    Parameters:
    directory_path (str): The path to the directory containing the files to merge.
    output_file_path (str): The path to the output file where the merged content will be written.
    """
    # Open the output file in write mode
    with open(output_file_path, 'w') as output_file:
        # Iterate over all the items in the directory
        for item in os.listdir(directory_path):
            # Construct the full path of the item
            item_path = os.path.join(directory_path, item)
            # Check if the item is a file
            if os.path.isfile(item_path):
                # Open the file in read mode
                with open(item_path, 'r') as input_file:
                    # Read the content of the file
                    file_content = input_file.read()
                    # Write the content to the output file
                    output_file.write(file_content + "\n")


def create_cross_validation_folders(source_folder,dest_folder, n_folds):
    # Check if the source folder exists
    if not os.path.isdir(source_folder):
        raise ValueError("Source folder does not exist.")
    create_path_if_not_exists(dest_folder)
    # Get all file names in the folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Shuffle the files to ensure randomness
    random.shuffle(all_files)

    # Split the files into n_folds parts
    fold_size = len(all_files) // n_folds
    folds = [all_files[i:i + fold_size] for i in range(0, len(all_files), fold_size)]

    # Ensure that all files are included (due to integer division)
    for i in range(len(all_files) - fold_size * n_folds):
        folds[i].append(all_files[-i - 1])

    # Create train and test folders for each fold
    for i, test_files in enumerate(folds):
        train_files = [f for fold in folds if fold != test_files for f in fold]

        fold_dir = os.path.join(dest_folder, f'fold_{i + 1}')
        train_dir = os.path.join(fold_dir, 'train')
        test_dir = os.path.join(fold_dir, 'test')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Copy files to train and test directories
        for f in train_files:
            shutil.copy(os.path.join(source_folder, f), train_dir)
        for f in test_files:
            shutil.copy(os.path.join(source_folder, f), test_dir)


def split_file_by_ratio(original_file_path, ratio, new_file_path):
    """
    Splits the content of the original file into two files based on a given ratio. The original file is modified
    to contain only a certain ratio of the total lines, and the remaining lines are written to a new file.

    Parameters:
    original_file_path (str): The path to the original file.
    ratio (float): The ratio of lines to keep in the original file (e.g., 0.5 for half the lines).
    new_file_path (str): The path to the new file where the exceeding lines will be written.
    """
    # Open the original file and read its lines
    with open(original_file_path, 'r') as file:
        lines = file.readlines()

    # Calculate the number of lines to keep based on the ratio
    line_count = int(len(lines) * ratio)

    # Lines to keep in the original file
    lines_to_keep = lines[:line_count]
    # Lines to move to the new file
    lines_to_move = lines[line_count:]

    # Write the lines to keep back into the original file
    with open(original_file_path, 'w') as file:
        file.writelines(lines_to_keep)

    # Write the exceeding lines to the new file
    with open(new_file_path, 'w') as file:
        file.writelines(lines_to_move)


def building_paths(fold):
    in_fold_train = cf.CROSS_ROOT_STD + fold + cf.XES_TRAIN_CROSS_SRC
    in_fold_test = cf.CROSS_ROOT_STD + fold + cf.XES_TEST_CROSS_SRC
    out_fold_train = cf.CROSS_ROOT_STD + fold + cf.XES_TRAIN_CROSS_DST
    out_fold_test = cf.CROSS_ROOT_STD + fold + cf.XES_TEST_CROSS_DST
    out_fold_gt = cf.CROSS_ROOT_STD + fold + cf.XES_GT_CROSS_DST

    # Ensure paths exist
    create_path_if_not_exists(in_fold_train)
    create_path_if_not_exists(in_fold_test)
    create_path_if_not_exists(out_fold_train)
    create_path_if_not_exists(out_fold_test)
    create_path_if_not_exists(out_fold_gt)

    return in_fold_train, in_fold_test, out_fold_train, out_fold_test, out_fold_gt

