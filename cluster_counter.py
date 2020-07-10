import ast
import copy
import statistics
import Evaluator as ev
import PostProcessing as pp
import text_to_model_output as ttmo
import statistics

tags = ['(I)', '(O)', '(P)', '(C)']
num_tags = 4
num_measures = 1 + 3*(num_tags - 2)

evaluator = ev.Evaluator(num_tags, num_measures, tags)
postprocessing = pp.PostProcessing(num_tags, tags)

def cluster_counter(unencodedY, y_test_dist, y_pred_class, y_pred_dist):
    (y_pred_class, c_pred_dist) = postprocessing.correct_dist_prediction(y_pred_class, y_pred_dist, unencodedY)
    (true_spans, pred_spans) = postprocessing.replace_argument_tag(y_pred_class, unencodedY)

    true_spans_dict_list = evaluator.spanCreator(true_spans)

    true_spans_closure = evaluator.edge_closure(true_spans, true_spans_dict_list, y_test_dist)

    true_edges = evaluator.remove_redundant_edges(true_spans_closure, true_spans, 'true')

    cluster_counter = []
    cluster_sizes = []
    for text in range(0, len(true_edges)):
        cluster_sets = []
        cluster_size = []
        for node, links in true_edges[text].items():
            new_cluster = set(links)
            new_cluster.add(node)
            added = False
            for cluster in cluster_sets:
                for link in links:
                    if link in cluster:
                        cluster = cluster.union(new_cluster)
                        added = True
                        break

            if not added:
                cluster_sets.append(new_cluster)
                cluster_size.append(len(new_cluster))

        cluster_counter.append(len(cluster_sets))
        cluster_sizes.append(statistics.mean(cluster_size))

    max_clusters = max(cluster_counter)
    min_clusters = min(cluster_counter)
    mean_clusters = statistics.mean(cluster_counter)
    stdev_clusters = statistics.stdev(cluster_counter)
    median_clusters = statistics.median(cluster_counter)

    print('clusters per file')
    print('mean: ', mean_clusters)
    print('max: ', max_clusters)
    print('min: ', min_clusters)
    print('stdev: ', stdev_clusters)
    print('median:', median_clusters)

    print('avg cluster size: ', statistics.mean(cluster_sizes))

(unencodedY, y_test_dist) = ttmo.main('pt', 'true')
(y_pred_class, y_pred_dist) = ttmo.main('pt', 'pred')
# print('ENGLISH')
print('PORTUGUESE')
cluster_counter(unencodedY, y_test_dist, y_pred_class, y_pred_dist)
