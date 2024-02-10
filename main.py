from GNN_engine import get_recommendations, eval_recommendations
import os
import dataset_utilities as du

import config as cf

import csv



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
        du.create_path_if_not_exists(cf.XES_SESSION_TRAIN_DST)
        preprocessed_train,train_data = du.encoding_training_data_dump(cf.XES_SESSION_TRAIN_DST + cf.KB_TRAIN)
        for file in os.listdir(cf.XES_TEST_DST):
            get_recommendations(train_preprocessed=preprocessed_train, train_data=train_data, test_context=cf.XES_TEST_DST+file,
                                  n_items = 10)

    elif type == "4":
        #du.create_cross_validation_folders(cf.XES_TRAIN_SRC)

        results_csv_path = 'results.csv'

        # Open the CSV file in append mode, so we don't overwrite existing data
        with open(results_csv_path, mode='a', newline='') as file:
            # Create a CSV writer object
            csv_writer = csv.writer(file)

            # Optionally, write headers if the file is newly created or empty
            # Uncomment the next line if you want to write headers
            # csv_writer.writerow(['Fold', 'Avg Precision', 'Avg Recall', 'Avg F1'])

            for fold in os.listdir(cf.CROSS_ROOT):
                # Construct paths
                in_fold_train = cf.CROSS_ROOT + fold + cf.XES_TRAIN_CROSS_SRC
                in_fold_test = cf.CROSS_ROOT + fold + cf.XES_TEST_CROSS_SRC
                out_fold_train = cf.CROSS_ROOT + fold + cf.XES_TRAIN_CROSS_DST
                out_fold_test = cf.CROSS_ROOT + fold + cf.XES_TEST_CROSS_DST
                out_fold_gt = cf.CROSS_ROOT + fold + cf.XES_GT_CROSS_DST

                # Ensure paths exist
                du.create_path_if_not_exists(in_fold_train)
                du.create_path_if_not_exists(in_fold_test)
                du.create_path_if_not_exists(out_fold_train)
                du.create_path_if_not_exists(out_fold_test)
                du.create_path_if_not_exists(out_fold_gt)

                # Preprocess and encode data
                preprocessed_train, train_data = du.encoding_training_data_dump(cf.CROSS_ROOT + fold + cf.CROSS_KB_SRC)

                # Initialize metrics accumulators
                total_precision = 0
                total_recall = 0
                total_f1 = 0
                file_count = 0

                # Iterate through each test file
                for file in os.listdir(out_fold_test):
                    pr, rec, f1 = eval_recommendations(train_preprocessed=preprocessed_train, train_data=train_data,
                                                       test_context=out_fold_test + file, gt_context=out_fold_gt + file,
                                                       n_items=10)

                    # Accumulate metrics
                    total_precision += pr
                    total_recall += rec
                    total_f1 += f1
                    file_count += 1

                # Calculate and write average metrics for the fold to the CSV file
                if file_count > 0:
                    avg_precision = total_precision / file_count
                    avg_recall = total_recall / file_count
                    avg_f1 = total_f1 / file_count
                    # Write the fold and its average metrics to the CSV file
                    csv_writer.writerow([fold, avg_precision, avg_recall, avg_f1])
                else:
                    print(f"Fold {fold}: No files to process.")


        return

if __name__ == "__main__":
    print("Select configuration:\n1.train \n2.recommendation from KB \n3.recommendation from last session \n4.cross fold validation")
    conf = input("Insert configuration: ")
    main(conf)


