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
    labels = []
    docs = []
    try:
        with open(filename,'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                if line.find('\t')!= -1:
                    content = line.split('\t')
                    labels.append(content[0])
                    docs.append(content[1].split(' ')[0])
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




def create_path_if_not(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_xes_traces(in_path, out_path, is_train):
    create_path_if_not(out_path)
    if is_train:
        with open(f"{out_path}train.txt", 'w', encoding='utf8', errors='ignore') as res:
            for file in os.listdir(in_path):
                try:
                    tree = ET.parse(in_path + file)
                    root = tree.getroot()
                    for trace in root.findall('trace'):
                        if len(trace) > 0:
                            for event in trace:
                                res.write("event" + '\t')
                                for attributes in event:
                                    if attributes.attrib.get('key') == "class":
                                        res.write(attributes.attrib.get('value') + " ")
                                    if attributes.attrib.get('key') == "featureName":
                                        res.write(attributes.attrib.get('value') + " ")
                                    if attributes.attrib.get('key') == "eventType":
                                        res.write(attributes.attrib.get('value') + "")
                                res.write("\n")
                except:
                    print("No log trace in file", file)
    else:
        for file in os.listdir(in_path):
            with open(f"{out_path}{file}", 'w', encoding='utf8', errors='ignore') as res:
                try:
                    tree = ET.parse(in_path + file)
                    root = tree.getroot()
                    for trace in root.findall('trace'):
                        if len(trace) > 0:
                            for event in trace:
                                res.write("event" + '\t')
                                for attributes in event:
                                    if attributes.attrib.get('key') == "class":
                                        res.write(attributes.attrib.get('value') + " ")
                                    if attributes.attrib.get('key') == "featureName":
                                        res.write(attributes.attrib.get('value') + " ")
                                    if attributes.attrib.get('key') == "eventType":
                                        res.write(attributes.attrib.get('value') + "")
                                res.write("\n")
                except:
                    print("No log trace in file", file)






def aggregate_cluster_files(path, outpath, filename):
    with open(outpath + filename, 'wb') as wfd:
        for f in os.listdir(path):
            try:
                with open(path + f, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
            except:
                continue








