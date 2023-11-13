from GNN_engine import get_recommendations
import os
import dataset_utilities as du

import config as cf

def main(type):
    '''
    - type 1 preparing data for training
        input: folder of XES models to train the system
        output: a single file with all classes and their operation
    - type 2: run the recommender on the knowledge base
        input: the train files and the list of test XES traces
        output: recommendations as string
    - type 3: run the recommender with increased training set
        input: MER folder and test models
        output: session-based recommendations
    '''
    if type == "1":
        du.parse_xes_traces(cf.XES_TRAIN_SRC, cf.XES_TRAIN_DST, True)
        du.encoding_training_data_dump(cf.XES_TRAIN_DST+cf.KB_TRAIN)
        return
    elif type == "2":
        du.parse_xes_traces(cf.XES_TEST_SRC, cf.XES_TEST_DST, False)
        preprocessed_train = du.load_preprocessed_data(cf.DUMP_TRAIN)
        train_data = du.load_file(cf.XES_TRAIN_DST + cf.KB_TRAIN)
        for file in os.listdir(cf.XES_TEST_DST):
            get_recommendations(train_preprocessed=preprocessed_train, train_data=train_data, test_context=cf.XES_TEST_DST+file,
                                  n_items = 10)
        return
    elif type == "3":
        du.parse_xes_traces(cf.XES_SESSION_TRAIN_SRC, cf.XES_SESSION_TRAIN_DST, True)
        du.create_path_if_not(cf.XES_SESSION_TRAIN_DST)
        preprocessed_train,train_data = du.encoding_training_data_dump(cf.XES_SESSION_TRAIN_DST + cf.KB_TRAIN)
        for file in os.listdir(cf.XES_TEST_DST):
            get_recommendations(train_preprocessed=preprocessed_train, train_data=train_data, test_context=cf.XES_TEST_DST+file,
                                  n_items = 10)
        return

if __name__ == "__main__":
    print("Select configuration:\n1.train \n2.recommendation from KB \n3.recommendation from last session")
    conf = input("Insert configuration: ")
    main(conf)


