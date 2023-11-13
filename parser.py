import xml.etree.ElementTree as ET
import os
import shutil
import pandas as pd
from collections import defaultdict
import numpy as np


def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s


def parse_json_file(path, out_path):
    for file in os.listdir(path):
        try:
            d = defaultdict(list)
            with open(path + file, 'r', encoding='utf-8', errors='ignore') as model_file:
                lines = model_file.readlines()
                # print(len(lines))
                for l in lines:
                    terms = l.split(' ')
                    for t in terms:
                        splitted = t.split('.')
                        if splitted[0] and splitted[1]:
                            # print(splitted[0], splitted[1])
                            d.setdefault(splitted[0], []).append(splitted[1])

            with open(out_path + file, 'w', encoding='utf-8') as results:
                for k, value in d.items():
                    results.write('label' + '\t' + k + ' ' + ' '.join(map(str, value)) + '\n')
        except:
            print(file)
            continue


def delete_empty_files(path):
    for file in os.listdir(path):
        if os.stat(path + file).st_size == 0:
            shutil.move(path + file, 'C:/Users/claudio/Desktop/empty_files')


def remove_small_models(in_path, out_path):
    for file in os.listdir(in_path):
        try:
            f = open(in_path + file, 'r', encoding='utf-8', errors='ignore')

            if len(f.readlines()) >= 20:
                f.close()
                shutil.move(in_path + file, out_path + file)
        except:
            print(file)
            continue

        # print(len(f.readlines()))


def import_fibo(file):
    df_fibo = pd.read_csv(file)
    dict_fibo = {}

    for term, synon in zip(df_fibo['Term'], df_fibo['Synonyms']):
        if str(synon) != 'nan':
            dict_fibo.update({term: synon})

    # for key, value in dict_fibo.items():
    #     print(key, value.split(','))

    return dict_fibo


def parse_utf_16_encoding():
    in_path = 'C:/Users/claudio/Desktop/ZapDev_dataset/utf-16_models/'
    out_path = 'C:/Users/claudio/Desktop/Zap_dev_new/round1/train/'

    list_classes = []
    list_attrib = []

    for file in os.listdir(in_path + '/'):
        try:
            # tree = ET.parse(in_path + folder + '/BusinessObjects/' + file)

            f = open(in_path + '/' + file, 'r')
            # print(f.read().rstrip())
            # print(f)
            # root = tree.getroot()

            root = ET.fromstring(f.read().rstrip().encode('utf-16-le'))
            if not os.path.exists(out_path + '/'):
                os.mkdir(out_path + '/')
            with open(out_path + '/' + file.replace('.xml', '.txt'), 'w', encoding='utf8', errors='ignore') as res:
                association_list = root.find('Associations')

                dict_associations = {}
                if association_list:
                    for association in association_list:
                        # print(association.attrib.get('Class1'))
                        dict_associations.update({association.attrib.get('Role1'): (
                            association.attrib.get('Role2'), association.attrib.get('Multiplicity1'),
                            association.attrib.get('Multiplicity2'))})
                # classes = root.attrib
                for model in root.findall('SoftwareModel'):

                    # for cl in model.findall('Classes'):
                    #     # for c in cl:
                    #     #     print(c.attrib)
                    for data in model.findall('Datastores'):
                        for elem in data:
                            for d in elem:
                                for e in d:
                                    print(e.tag)
        except:

            print(file)


def parse_configuration(file, out_file):
    with open('features.txt', 'w', encoding='utf-8', errors='ignore') as res:
        try:
            tree = ET.parse(file)

            root = tree.getroot()
            for child in root:
                res.write('private boolean ' + str(child.attrib.get('name')) + ';\n')
        except:
            print("error")


def parser_xml():
    in_path = 'C:/Users/claudio/Desktop/K-fold_models/1/test/'
    out_path = 'C:/Users/claudio/Desktop/Zap_dev_new/round1/test/'

    list_classes = []
    list_attrib = []

    for file in os.listdir(in_path):
        try:
            tree = ET.parse(in_path + file)

            # f = open(in_path+folder+file, 'r')
            # print(f.read().rstrip())

            root = tree.getroot()

            # root=ET.fromstring(f.read().rstrip())
            if not os.path.exists(out_path + '/'):
                os.mkdir(out_path + '/')
            with open(out_path + '/' + file.replace('.xml', '.txt'), 'w', encoding='utf8', errors='ignore') as res:
                association_list = root.find('Associations')

                dict_associations = {}
                if association_list:
                    for association in association_list:
                        # print(association.attrib.get('Class1'))
                        dict_associations.update({association.attrib.get('Role1'): (
                            association.attrib.get('Role2'), association.attrib.get('Multiplicity1'),
                            association.attrib.get('Multiplicity2'))})
                classes = root.findall('Classes')
                if classes:
                    for root_class in classes:

                        for cl in root_class:
                            # print(cl.tag, cl.attrib)

                            res.write(str(cl.attrib.get('ModelName')) + '\t' + str(cl.attrib.get('Name')) + ' ')
                            list_classes.append(cl.attrib.get('Name'))
                            for dia in cl:
                                # print(dia.tag, dia.attrib)
                                for attr in dia:
                                    res.write('(' + str(attr.attrib.get('Name')) + ',' + str(
                                        attr.attrib.get('DataType')) + ') ')
                                    list_attrib.append(attr.attrib.get('Name'))

                            res.write(str(dict_associations.get(cl.attrib.get('Name'))))
                            res.write('\n')

        except:
            shutil.copy(in_path + file, 'C:/Users/claudio/Desktop/ZapDev_dataset/utf-16_models/' + file)
            print(file)

    for file in os.listdir(out_path):
        if not os.path.getsize(out_path + file):
            os.remove(out_path + file)

    return list_classes, list_attrib


def split_train_test_files(path, out_folder):
    for folder in os.listdir(path):
        count = 0
        number_files = os.listdir(path + folder)
        print(folder, len(number_files))

        num_tests = int(len(number_files) / 3)
        print(folder, num_tests)
        for file in number_files:
            if not os.path.exists(out_folder + folder):
                os.mkdir(out_folder + folder)
            shutil.move(path + folder + '/' + file, out_folder + folder + '/' + file)
            count += 1
            if count > num_tests:
                break


def split_test_files(root_path, n, filename, test, gt):
    from itertools import zip_longest

    def grouper(n, iterable, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    with open(root_path) as f:

        for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
            if i == 1:
                with open(test + filename, 'w') as fout:
                    fout.writelines(g)
            else:
                with open(gt + filename, 'w') as fout:
                    fout.writelines(g)


def split_train_clusters(root_path, n, filename, test):
    from itertools import zip_longest

    def grouper(n, iterable, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    with open(root_path) as f:
        for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
            with open(test + filename + '_{0}.txt'.format(i), 'w') as fout:
                fout.writelines(g)


def create_ten_fold_structure(path, out_folder):
    # splitted_path='./split_files/'
    # for fold in os.listdir(cat_path):

    # CommonFramework2.txt_1.txt
    for i in range(1, 11):
        for folder in os.listdir(path):
            for file in os.listdir(path + folder):
                filename = path + folder + '/' + file

                # if not os.path.exists(out_folder):
                #     os.mkdir(out_folder)
                # folder = out_folder + 'train_partial_'+str(i)+'/'
                print(filename)
                print(out_folder + file)
                try:
                    shutil.copy(filename, out_folder + file)
                except FileNotFoundError:
                    continue


def aggregate_cluster_files(path, outpath, filename):
    with open(outpath + filename, 'wb') as wfd:
        for f in os.listdir(path):
            try:
                with open(path + f, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
            except:
                continue


def create_clusters():
    filter_path = 'C:/Users/claudio/Desktop/ZapDev_dataset/clusters/'

    out_test_path = 'C:/Users/claudio/Desktop/ZapDev_dataset/train_zap/'

    for file in os.listdir(filter_path):
        with open(filter_path + file, 'r') as f:
            num = int(len(f.readlines()) / 10)
            split_train_clusters(filter_path + file, num, file, out_test_path)


def computes_avg_metrics(results_file):
    column_names = ['pr', 'rec', 'f1', 'succ', 'time']
    df_results = pd.read_csv(results_file, names=column_names)
    df_half = df_results.iloc[75:,:]
    avg_success = df_half['succ'].mean()
    avg_pr = df_half['pr'].mean()
    avg_rec = df_half['rec'].mean()
    avg_f1 = df_half['f1'].mean()
    avg_time = df_half['time'].mean()

    return avg_pr, avg_rec, avg_f1, avg_success, avg_time


def computes_avg_metrics_sim(results_file):
    df_results = pd.read_csv(results_file, names=['accuracy'])
    avg_success = df_results['accuracy'].mean()
    # avg_pr=df_results['precision'].mean()
    # avg_rec=df_results['recall'].mean()
    # avg_f1=df_results['f-measure'].mean()
    return avg_success


def extract_data():
    # dict_fibo=import_fibo('FIBO_development.csv')

    list_classes, list_attrib = parser_xml()

    with open('classes.txt', 'w', encoding='utf8', errors='ignore') as class_file:
        for cl in set(list_classes):
            class_file.write(cl + '\n')
    with open('attributes.txt', 'w', encoding='utf8', errors='ignore') as attr_file:
        for attr in set(list_attrib):
            attr_file.write(attr + '\n')


# parser_xml()
# parse_utf_16_encoding()

# parse_utf_16_encoding()
# for cl in set(list_classes):
#     if dict_fibo.get(str(cl).lower()):
#         print('class '+cl,dict_fibo.get(str(cl).lower()))
#
# for attr in set(list_attrib):
#     if dict_fibo.get(str(attr).lower()):
#         print('attrib '+attr,dict_fibo.get(str(attr).lower()))
#
# path_train='C:/Users/claudio/Desktop/ten_folder_modelSet/'
# out_path = 'C:/Users/claudio/Desktop/ten_folder_modelSet/'
# for folder in os.listdir(path_train):
#     aggregate_cluster_files(path_train+folder+'/',out_path, folder+'.txt')


# create_clusters()

# train_path='C:/Users/claudio/Desktop/ZapDev_dataset/train_zap/'
# out_path_clusters='C:/Users/claudio/Desktop/ZapDev_dataset/train_structure/'
# for i in range(1,11):
#     for file in os.listdir(train_path):
#         print(file)
#         if str(file).find('_'+str(i)) != -1:
#             print(file)
#             shutil.copy(train_path+file, out_path_clusters+'train_partial_'+str(i)+'/'+file)
#
#         else:
#             print('none')


# split_train_test_files('C:/Users/claudio/Desktop/ZapDev_dataset/morgan_format/','C:/Users/claudio/Desktop/ZapDev_dataset/test_files/')
# create_ten_fold_structure('C:/Users/claudio/Desktop/ZapDev_dataset/morgan_format/', 'C:/Users/claudio/Desktop/ZapDev_dataset/ten_folder_zap/root/')
# create_ten_fold_structure(dest_path)
# for i in range(1, 11):
#     fold_path = 'C:/Users/claudio/Desktop/ZapDev_dataset/train_structure/train_partial_' + str(i) + '/'
#     out_path = 'C:/Users/claudio/Desktop/ZapDev_dataset/train_structure/train_main/'
#     filename = 'train_partial_' + str(i) + '.txt'
#     aggregate_cluster_files(fold_path, out_path, filename)

# for i in range (1,11):
#     cluster_path ='C:/Users/claudio/Desktop/ten_fold_ecore_structure/test'+str(i)+'/'
#     filter_path = './test_categories/test_'+str(i)+'/'
#     split_path = './split_files/test_'+str(i)+'/'
#     out_gt_path = 'C:/Users/claudio/Desktop/test_classes/gt_'+str(i)+'/'
#     out_test_path = 'C:/Users/claudio/Desktop/test_classes/test_'+str(i)+'/'

#


# #
# round= 1

def split_test_gt_files(path):
    for i in range(1, 11):
        gt_path = path + 'gt' + str(i) + '/'
        test_path = path + 'test_partial' + str(i) + '/'
        if not os.path.exists(gt_path):
            os.mkdir(gt_path)
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        root = path + 'test' + str(i) + '/'
        for file in os.listdir(root):
            try:
                with open(root + file, 'r') as f:
                    num = int(len(f.readlines()) / 2)
                    split_test_files(root + file, num, file, test_path, gt_path)
            except:
                print(file)
                continue


def compute_metrics(path):
    sum_succ = 0
    sum_pr = 0
    sum_rec = 0
    sum_f1 = 0
    sum_time = 0

    cosine_sum_succ = 0
    cosine_sum_pr = 0
    cosine_sum_rec = 0
    cosine_sum_f1 = 0

    lev_sum_succ = 0
    lev_sum_pr = 0
    lev_sum_rec = 0
    lev_sum_f1 = 0

    for i in range(1, 11):
        pr, rec, f1, succ, time = computes_avg_metrics(path + '\\results_round' + str(i) + '.csv')

        print(pr, rec, f1, succ, time)
        sum_succ += succ
        sum_pr += pr
        sum_rec += rec
        sum_f1 += f1
        sum_time += time

        # cosine_sum_succ += succ_cosine
        # cosine_sum_pr += pr_cosine
        # cosine_sum_rec += rec_cosine
        # cosine_sum_f1 += f1_cosine
        #
        # lev_sum_succ += succ_lev
        # lev_sum_pr += pr_lev
        # lev_sum_rec += rec_lev
        # lev_sum_f1 += f1_lev

    print('std metrics')
    print(sum_succ / 10)
    print(sum_pr / 10)
    print(sum_rec / 10)
    print(sum_f1 / 10)
    print(sum_time / 10)

    # print('='*60)
    # print('cosine metric')
    # print(cosine_sum_succ/10)
    # print(cosine_sum_pr/10)
    # print(cosine_sum_rec/10)
    # print(cosine_sum_f1/10)
    #
    # print('='*60)
    # print('lev metric')
    # print(lev_sum_succ/10)
    # print(lev_sum_pr/10)
    # print(lev_sum_rec/10)
    # print(lev_sum_f1/10)


def compute_map_metric(df_metric):
    df_metric.columns = ['pr_std', 'rec_std', 'f1_std', 'succ_std']
    # df_metric = pd.read_csv(csv_results, sep=',', names=column_names)

    precisions_std = df_metric['pr_std'].values.tolist()
    recalls_std = df_metric['rec_std'].values.tolist()

    print(type(recalls_std))

    precisions_std.append(100)
    recalls_std.append(0)
    # print(recalls_std)
    precisions_std = np.array(precisions_std)
    recalls_std = np.array(recalls_std)
    precisions_std_scaled = precisions_std / 100
    recalls_std_scaled = recalls_std / 100
    # print(recalls_std)
    ap_std = np.sum((recalls_std_scaled[:-1] - recalls_std_scaled[1:]) * precisions_std_scaled[:-1])
    print('map std :', ap_std / len(recalls_std))

    ## cosine ##
    # precisions_cos = df_metric['pr_cosine'].values.tolist()
    # recalls_cos = df_metric['rec_cosine'].values.tolist()
    # precisions_cos.append(1)
    # recalls_cos.append(0)
    #
    # precisions_cos = np.array(precisions_cos)
    # recalls_cos = np.array(recalls_cos)
    #
    # ap_cos = np.sum((recalls_cos[:-1] - recalls_cos[1:]) * precisions_cos[:-1])
    # print('map cos :', ap_cos )
    # ## lev ##
    # precisions_lev = df_metric['pr_lev'].values.tolist()
    # recalls_lev = df_metric['rec_lev'].values.tolist()
    # precisions_lev.append(1)
    # recalls_lev.append(0)
    #
    # precisions_lev = np.array(precisions_lev)
    # recalls_lev = np.array(recalls_lev)
    # ap_lev = np.sum((recalls_lev[:-1] - recalls_lev[1:]) * precisions_lev[:-1])
    #
    # print('map lev :', ap_lev)


def remove_duplicates(in_file, out_file):
    uniqlines = set(open(in_file).readlines())
    no_dup_file = open(out_file, 'w').writelines(uniqlines)


# remove_duplicates('C:/Users/claudio/PycharmProjects/Grakel/Datasets/D_beta/C2.1/train2.txt', 'C:/Users/claudio/PycharmProjects/Grakel/Datasets/D_beta/C2.1/train2_no_dup.txt')
# compute_map_metric('C:/Users/claudio/Desktop/Morgan_extension_results/Ecore/classes/results_round1.csv')

# i=1
def parse_xse_files(in_path, out_path):
    for file in os.listdir(in_path):

        tree = ET.parse(in_path + file)

        root = tree.getroot()

        # root=ET.fromstring(f.read().rstrip())
        if not os.path.exists(out_path + '/'):
            os.mkdir(out_path + '/')
        with open(out_path + '/' + file.replace('.xmi', '.txt'), 'w', encoding='utf8', errors='ignore') as res:
            # print("root ", root.attrib)

            for event in root:

                # print(trace.attrib)

                print(event.attrib)

                if event.attrib.get('class'):
                    res.write("trace" + '\t')
                    res.write(event.attrib.get('class') + ' ')
                if event.attrib.get('feature'):
                    res.write(event.attrib.get('feature'))
                    res.write('\n')


def move_files(root, train_path, test_path, num_test, flag, flag_test):
    i = 0

    if flag:
        set_dir = root

    if not flag and flag_test:
        set_dir = root

    if not flag_test and not flag:
        set_dir = test_path

    for file in sorted(os.listdir(set_dir), key=lambda v: v.upper()):
        if flag:
            try:
                shutil.move(root + file, test_path + file)
                i = i + 1
                if i == num_test:
                    break
            except:
                print(file)
                continue

        else:
            try:
                if flag_test:
                    shutil.copy(root + file, train_path + file)
                else:
                    shutil.copy(test_path + file, train_path + file)
            except:
                print(file)
                continue


def merge_ten_folds(src):
    list_df = []
    for i in range(1, 11):
        path = src + 'results_round' + str(i) + '.csv'

        # column_names = ['pr', 'rec', 'f1', 'succ']
        df_results = pd.read_csv(path)
        print(df_results)

        list_df.append(df_results)

    df_merged = pd.concat(list_df)
    # print(df_merged.describe())
    # df_merged.to_csv('C:\\Users\\claud\\OneDrive\\Desktop\\Morken_comparison\\D_beta_comparison\\D_beta_classes_std.csv',index=False)
    return df_merged


def remove_duplicates_from_dataframe(csv_file, path, outfile):
    df_data = pd.read_csv(csv_file, sep=',')
    no_dup = df_data.drop_duplicates()
    no_dup.to_csv(path + outfile, index=False)
    return no_dup


def convert_string_list_to_int(string_list):
    return [eval(i) for i in string_list]


def average_int(lst):
    return sum(lst) / len(lst)


def export_recommendation_to_xes(file_path, rec_list):
    ## get root from existing model
    context = ET.parse(file_path)

    root = context.getroot()
    new_root = ET.Element("root")
    # new_root = ET.Element('xmi:XMI xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:ecoreXES="http://www.example.org/ecoreXES')
    print(root)
    for trace in root:
        rec = ET.Element("event")
        rec.set("eventtype", "delete")
        trace.append(rec)
    context.write('output.xmi')
    # doc = ET.SubElement(new_root, "doc")
    #
    # ET.SubElement(doc, "field1", name="blah").text = "some value1"
    # ET.SubElement(doc, "field2", name="asdfasd").text = "some vlaue2"
    #
    # tree = ET.ElementTree(new_root)



# # #src='C:/Users/claudio/Desktop/mnb_train/root/'
# tot = len(os.listdir(ten_folder_path+'root/'))



# print(test)
# train = tot - int(test)
#
# count = len(os.listdir(ten_folder_path+'root/'))
#


def create_ten_folders(path):
    for i in range(1, 11):
        os.mkdir(path + "train" + str(i))
        os.mkdir(path + "test" + str(i))


def run_ten_folder(ten_folder_path):
    test = 85
    for actual_round in range(1, 11):

        previous_round = range(1, actual_round)

        move_files(ten_folder_path + 'root/', ten_folder_path + 'train' + str(actual_round) + '/',
                   ten_folder_path + 'test' + str(actual_round) + '/', test, True, False)

        move_files(ten_folder_path + 'root/', ten_folder_path + 'train' + str(actual_round) + '/',
                   ten_folder_path + 'test' + str(actual_round) + '/', test, False, True)

        for i in previous_round:
            move_files(ten_folder_path + 'root/', ten_folder_path + 'train' + str(actual_round) + '/',
                       ten_folder_path + 'test' + str(i) + '/', test, False, False)


# compute_metrics()


# outpath= "C:/Users/claud/OneDrive/Desktop/GitRanking_results/topics_lv8/ten_folder_rounds/"
# with open(outpath+'results_lv8_avg.csv', 'w', encoding='utf-8', errors='ignore') as res:
#     for i in range(1,11):
#         print('round', i)
#         succ, pr, rec, f1 = computes_avg_metrics(outpath+"results_lv8_round_"+str(i)+".csv")
#         res.write(str(succ)+','+str(pr)+','+str(rec)+','+str(f1)+'\n')

# print(computes_avg_metrics('C:/Users/claudio/Desktop/Morgan_extension_results/Ecore/classes_ontology/results_round1.csv'))
# remove_small_models('D:/backup_datasets/MORGAN_extension/parsed_json/', 'D:/backup_datasets/MORGAN_extension/discarded/')
# delete_empty_files('C:/Users/claudio/Desktop/parsed_json/')
# parse_json_file('C:/Users/claudio/Desktop/BI/', 'C:/Users/claudio/Desktop/parsed_json/')
# print(computes_avg_metrics("C:/Users/claud/Desktop/Grakel/Grakel/Datasets/D_beta/C2.1/results_round2.csv"))
# ten_folder_pipeline()
# compute_metrics()
# parse_xse_files('./Datasets/XES_new_dataset/','./Datasets/results_XMI/')

# rec_list =['rec1', 'another rec']
# export_recommendation_to_xes("C:/Users/claud/OneDrive/Desktop/Grakel/Grakel/Datasets/ecoreXES/result0_100.xmi",rec_list)
# remove_duplicates("C:/Users/claud/Desktop/Morgan_json_dataset/Morgan_json_dataset/train_10.txt","C:/Users/claud/Desktop/Morgan_json_dataset/Morgan_json_dataset/train_10_no_dup.txt")
# parse_configuration('C:/Users/claudio/Desktop/KNN.xml', 'features.txt')


# path="C:/Users/claud/OneDrive/Desktop/Dati_lavoro/MORGAN_BORA/"
# outfile = "graph_metrics_ecore_no_dup.csv"
# remove_duplicates_from_dataframe(path+"graph_metrics_ecore.csv", path, outfile)

# df_zap = pd.read_csv("C:/Users/claud/OneDrive/Desktop/Dati_lavoro/MORGAN_BORA/graph_metrics_zapDev_no_dup.csv")
# df_ecore = pd.read_csv("C:/Users/claud/OneDrive/Desktop/Dati_lavoro/MORGAN_BORA/graph_metrics_ecore_no_dup.csv")
# df_cdm = pd.read_csv("C:/Users/claud/OneDrive/Desktop/Dati_lavoro/MORGAN_BORA/graph_metrics_cdm_no_dup.csv")
#
# print("avg_size_ecore: ", average_int(list(df_ecore['graph_size'].values)))
# print("avg_size_zapDev: ", average_int(list(df_zap['graph_size'].values)))
# print("avg_size_cdm: ", average_int(list(df_cdm['graph_size'].values)))
#
# print("avg_order_ecore: ", average_int(list(df_ecore['graph_order'].values)))
# print("avg_order_zapDev: ", average_int(list(df_zap['graph_order'].values)))
# print("avg_order_cdm: ", average_int(list(df_cdm['graph_order'].values)))
#
# print("avg_density_ecore: ", average_int(list(df_ecore['density'].values)))
# print("avg_density_zapDev: ", average_int(list(df_zap['density'].values)))
# print("avg_density_cdm: ", average_int(list(df_cdm['density'].values)))

# avg_size_ecore = average(convert_string_list_to_int(ecore_sizes))

# print(df_ecore['graph_size'].mean())

# for i in range(1,11):
#     df_merged= pd.read_csv("C:/Users/claud/OneDrive/Desktop/Dati_lavoro/MORGAN_BORA/CDM_classes/results_round"+str(i)+".csv")
#     compute_map_metric(df_merged)



compute_metrics("C:\\Users\\claud\\OneDrive\\Desktop\\Dati_lavoro\\Morgan_sosym\\Revision_2nd\\results_time_uml\\")




#create_ten_folders("C:\\Users\\claud\\OneDrive\\Desktop\\Dati_lavoro\\Morgan_sosym\\Revision_2nd\\parsed_uml\\")
#ten_folder_path = 'C:/Users/claud/OneDrive/Desktop/Dati_lavoro/Morgan_sosym/Revision_2nd/parsed_uml/'



# for i in range(1,11):
#    aggregate_cluster_files(path=ten_folder_path+"train"+str(i)+"/", outpath=ten_folder_path,filename='train'+str(i)+'.txt')

#split_test_gt_files(ten_folder_path)
#run_ten_folder(ten_folder_path)