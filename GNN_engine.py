import os
import time


from grakel.utils import graph_from_networkx

from custom_kernel_matrix import CustomKernelMatrix
import dataset_utilities as du
import config as cf



def compute_recommendations(G_train, train_data, G_test, n):
    ranked_list = ()
    list_sim = []
    for g, rec in zip(G_train, train_data):
        try:
            if len(G_test[n]) > 0:
                sim = compute_kernel_similarity(g, G_test[n])
                if sim[0][0] > 0:
                    tuple_g = rec, sim[0][0]
                    list_sim.append(tuple_g)
                ranked_list = sorted(list_sim, key=lambda tup: tup[1], reverse=True)
            else:
                continue
        except IndexError:
            continue

    return set(ranked_list)



def compute_kernel_similarity(g_train,g_test):
    sp_kernel = CustomKernelMatrix()
    sp_kernel.fit_transform([g_train])
    sp_kernel.transform([g_test])
    return sp_kernel.transform([g_test])



def join_rec(dict_results):
    cut_rec = dict_results
    combined_list = []
    for elem in cut_rec:
        rec_graph = elem[0].split(' ')

        combined_list.extend(rec_graph[0:1])


    return combined_list




def get_recommendations(train_preprocessed, train_data,test_context,n_items):

    with open(test_context, 'r', errors='ignore', encoding='utf-8') as f:
        lenght = len(f.readlines())
        test_preprocessed, test_data = du.encoding_data(test_context)

        vocab = du.get_vocab(train_preprocessed, test_preprocessed)
        G_train_nx = du.create_graphs_of_words(train_preprocessed, vocab, 3)
        G_test_nx = du.create_graphs_of_words(test_preprocessed, vocab, 3)
        G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='label'))
        G_test = list(graph_from_networkx(G_test_nx, node_labels_tag='label'))
        start = time.time()

        for i in range(0, lenght):
            results = compute_recommendations(G_train, train_data, G_test, i)
            rec_graph = join_rec(results)
        end = time.time()
        enlapsed = end - start

        print("Rec time: ", enlapsed)
        gt_data = du.get_gt_classes(test_context)

        if gt_data:
            rec_graph = set(rec_graph)
            rec_graph = list(rec_graph)[0:n_items]

            list_gt_global = gt_data

            print('recommended operations ', rec_graph)
            if list_gt_global:
                operations = du.match_operations(rec_graph,list_gt_global,test_data)

            produce_recommendations_dump(operations, test_context)





def produce_recommendations_dump(recommendations, test):
    du.create_path_if_not(cf.REC_DST)
    head, tail = os.path.split(test)
    #out_recs = []
    with open(f"{cf.REC_DST}/recommendations_for_{tail}", 'w') as res:
        for key, value in recommendations.items():
            out_string = value[2].strip() + " attribute" + key + " to class " + value[1]
            #out_recs.append(out_string)
            res.write(f"{out_string}\n")
            print(out_string)


# TODO: visual recommendation
def show_recommendations(recommendations, template_path):
    return
    # environment = Environment(loader=FileSystemLoader(template_path))
    # template = environment.get_template("recommendation_template.txt")
    #
    # for key,value in recommendations.items():
    #     filename = f"recommendation_for_{test_file}.txt"
    #     content = template.render(
    #         list_rec=recommendations,
    #          attr_name=key,
    #         class_name = value[1],
    #         type_op= value[2]
    #     )
    #
    #
    #     with open(out_path+filename, mode="w", encoding="utf-8") as message:
    #         message.write(content)
    #         print(f"... wrote {filename}")

    # window = tk.Tk()
    # frame = tk.Frame(master=window, width=300, height=300)
    # frame.pack()
    # label = tk.Label(master=frame,  text='\n'.join(out_recs))
    # label.place(x=0, y=0)
    # label.pack()
    #
    # window.mainloop()
