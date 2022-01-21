# import pandas as pd

# gts = pd.DataFrame.from_dict([
#     {'query': 'q1', 'document': 'doc2'},
#     {'query': 'q1', 'document': 'doc3'},
#     {'query': 'q2', 'document': 'doc7'},
# ])

# results = pd.DataFrame.from_dict([
#     {'query': 'q1', 'document': 'doc1', 'rank': 1},
#     {'query': 'q1', 'document': 'doc2', 'rank': 2},
#     {'query': 'q1', 'document': 'doc3', 'rank': 3},
#     {'query': 'q2', 'document': 'doc4', 'rank': 1},
#     {'query': 'q2', 'document': 'doc5', 'rank': 2},
#     {'query': 'q2', 'document': 'doc6', 'rank': 3},
# ])

# MAX_RANK = 100000

# hits = pd.merge(gts, results, on=["query", "document"], how="left").fillna(MAX_RANK)

# mrr = (1 / hits.groupby('query')['rank'].min()).mean()

# print(mrr)

# import numpy as np


# def mean_reciprocal_rank(rs):
#     """Score is reciprocal of the rank of the first relevant item
#     First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
#     Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
#     >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
#     >>> mean_reciprocal_rank(rs)
#     0.61111111111111105
#     >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
#     >>> mean_reciprocal_rank(rs)
#     0.5
#     >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
#     >>> mean_reciprocal_rank(rs)
#     0.75
#     Args:
#         rs: Iterator of relevance scores (list or numpy) in rank order
#             (first element is the first item)
#     Returns:
#         Mean reciprocal rank
#     """
#     rs = (np.asarray(r).nonzero()[0] for r in rs)
#     return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


# def r_precision(r):
#     """Score is precision after all relevant documents have been retrieved
#     Relevance is binary (nonzero is relevant).
#     >>> r = [0, 0, 1]
#     >>> r_precision(r)
#     0.33333333333333331
#     >>> r = [0, 1, 0]
#     >>> r_precision(r)
#     0.5
#     >>> r = [1, 0, 0]
#     >>> r_precision(r)
#     1.0
#     Args:
#         r: Relevance scores (list or numpy) in rank order
#             (first element is the first item)
#     Returns:
#         R Precision
#     """
#     r = np.asarray(r) != 0
#     z = r.nonzero()[0]
#     if not z.size:
#         return 0.
#     return np.mean(r[:z[-1] + 1])


# def precision_at_k(r, k):
#     """Score is precision @ k
#     Relevance is binary (nonzero is relevant).
#     >>> r = [0, 0, 1]
#     >>> precision_at_k(r, 1)
#     0.0
#     >>> precision_at_k(r, 2)
#     0.0
#     >>> precision_at_k(r, 3)
#     0.33333333333333331
#     >>> precision_at_k(r, 4)
#     Traceback (most recent call last):
#         File "<stdin>", line 1, in ?
#     ValueError: Relevance score length < k
#     Args:
#         r: Relevance scores (list or numpy) in rank order
#             (first element is the first item)
#     Returns:
#         Precision @ k
#     Raises:
#         ValueError: len(r) must be >= k
#     """
#     assert k >= 1
#     r = np.asarray(r)[:k] != 0
#     if r.size != k:
#         raise ValueError('Relevance score length < k')
#     return np.mean(r)


# def average_precision(r):
#     """Score is average precision (area under PR curve)
#     Relevance is binary (nonzero is relevant).
#     >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
#     >>> delta_r = 1. / sum(r)
#     >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
#     0.7833333333333333
#     >>> average_precision(r)
#     0.78333333333333333
#     Args:
#         r: Relevance scores (list or numpy) in rank order
#             (first element is the first item)
#     Returns:
#         Average precision
#     """
#     r = np.asarray(r) != 0
#     out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
#     if not out:
#         return 0.
#     return np.mean(out)


# def mean_average_precision(rs):
#     """Score is mean average precision
#     Relevance is binary (nonzero is relevant).
#     >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
#     >>> mean_average_precision(rs)
#     0.78333333333333333
#     >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
#     >>> mean_average_precision(rs)
#     0.39166666666666666
#     Args:
#         rs: Iterator of relevance scores (list or numpy) in rank order
#             (first element is the first item)
#     Returns:
#         Mean average precision
#     """
#     return np.mean([average_precision(r) for r in rs])


# def dcg_at_k(r, k, method=0):
#     """Score is discounted cumulative gain (dcg)
#     Relevance is positive real values.  Can use binary
#     as the previous methods.
#     Example from
#     http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
#     >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
#     >>> dcg_at_k(r, 1)
#     3.0
#     >>> dcg_at_k(r, 1, method=1)
#     3.0
#     >>> dcg_at_k(r, 2)
#     5.0
#     >>> dcg_at_k(r, 2, method=1)
#     4.2618595071429155
#     >>> dcg_at_k(r, 10)
#     9.6051177391888114
#     >>> dcg_at_k(r, 11)
#     9.6051177391888114
#     Args:
#         r: Relevance scores (list or numpy) in rank order
#             (first element is the first item)
#         k: Number of results to consider
#         method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
#                 If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
#     Returns:
#         Discounted cumulative gain
#     """
#     r = np.asfarray(r)[:k]
#     if r.size:
#         if method == 0:
#             return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
#         elif method == 1:
#             return np.sum(r / np.log2(np.arange(2, r.size + 2)))
#         else:
#             raise ValueError('method must be 0 or 1.')
#     return 0.


# def ndcg_at_k(r, k, method=0):
#     """Score is normalized discounted cumulative gain (ndcg)
#     Relevance is positive real values.  Can use binary
#     as the previous methods.
#     Example from
#     http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
#     >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
#     >>> ndcg_at_k(r, 1)
#     1.0
#     >>> r = [2, 1, 2, 0]
#     >>> ndcg_at_k(r, 4)
#     0.9203032077642922
#     >>> ndcg_at_k(r, 4, method=1)
#     0.96519546960144276
#     >>> ndcg_at_k([0], 1)
#     0.0
#     >>> ndcg_at_k([1], 2)
#     1.0
#     Args:
#         r: Relevance scores (list or numpy) in rank order
#             (first element is the first item)
#         k: Number of results to consider
#         method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
#                 If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
#     Returns:
#         Normalized discounted cumulative gain
#     """
#     dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
#     if not dcg_max:
#         return 0.
#     return dcg_at_k(r, k, method) / dcg_max


# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()

# import numpy as np
# rng = np.random.RandomState(1)
# X = rng.randint(5, size=(6, 100))
# y = np.array([1, 2, 3, 4, 5, 6])
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# s = clf.fit(X, y)

# print(s.predict(X[2:3]))
# print(s)

lstp = [(1,2,3),(4,5,6),(7,8,9)]

# import pandas as pd
# import pandas
# import numpy as np
# import sys
# movies = pd.read_csv("movies.csv", encoding="utf-8")
# ratings = pd.read_csv("ratings.csv", encoding="utf-8")
# # print(movies)
# reciprocal_ranks = []
# indexes_of_fixes = np.flatnonzero(sorted_df['used_in_fix'] == 1.0)

# import numpy as np
# # rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

# rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])

# rs = (np.asarray(r).nonzero()[0] for r in rs)

# print(np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs]))


# import numpy as np

# __author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


# def precision_at_k(ranking, k):
#     """
#     Score is precision @ k
#     Relevance is binary (nonzero is relevant).
#     :param ranking: Relevance scores (list or numpy) in rank order (first element is the first item)
#     :type ranking: list, np.array
#     :param k: length of ranking
#     :type k: int
#     :return: Precision @ k
#     :rtype: float
#     """

#     assert k >= 1
#     ranking = np.asarray(ranking)[:k] != 0
#     if ranking.size != k:
#         raise ValueError('Relevance score length ({}) < k ({})'.format(ranking.size, k))
#     return np.mean(ranking)


# def average_precision(ranking):
#     """
#     Score is average precision (area under PR curve). Relevance is binary (nonzero is relevant).
#     :param ranking: Relevance scores (list or numpy) in rank order (first element is the first item)
#     :type ranking: list, np.array
#     :return: Average precision
#     :rtype: float
#     """

#     ranking = np.asarray(ranking) != 0
#     out = [precision_at_k(ranking, k + 1) for k in range(ranking.size) if ranking[k]]
#     if not out:
#         return 0.
#     return np.mean(out)


# def mean_average_precision(ranking):
#     """
#     Score is mean average precision. Relevance is binary (nonzero is relevant).
#     :param ranking: Relevance scores (list or numpy) in rank order (first element is the first item)
#     :type ranking: list, np.array
#     :return: Mean average precision
#     :rtype: float
#     """

#     return np.mean([average_precision(r) for r in ranking])


# def ndcg_at_k(ranking, k = None):
#     """
#     Score is normalized discounted cumulative gain (ndcg). Relevance is positive real values.  Can use binary
#     as the previous methods.
#     :param ranking: ranking to evaluate in dcg format [0, 0, 1], where 1 is correct info
#     :type ranking: list
#     :return: Normalized discounted cumulative gain
#     :rtype: float
#     """

#     k = len(ranking) if k is None else k 
#     assert k >= 1
#     # if ranking.size != k:
#     #     raise ValueError('Relevance score length ({}) < k ({})'.format(ranking.size, k))

#     ranking = np.asfarray(ranking)[:k] 
#     r_ideal = np.asfarray(sorted(ranking, reverse=True))
#     dcg_ideal = r_ideal[0] + np.sum(r_ideal[1:] / np.log2(np.arange(2, r_ideal.size + 1)))
#     dcg_ranking = ranking[0] + np.sum(ranking[1:] / np.log2(np.arange(2, ranking.size + 1)))

#     return dcg_ranking / dcg_ideal


# def reciprocal_rank(ranking, k=None):
#     """
#         Score is reciprocal of the rank of the first relevant item. 
#         First element is rank 1. Relevance is binary (nonzero is relevant).
#     :param ranking: Relevance scores (list or numpy) in rank order (first element is the first item)
#     :type ranking: list, np.array
#     :return: reciprocal rank of a ranking list
#     :rtype: float
#     """
#     k = len(ranking) if k is None else k
#     assert k >= 1

#     ranking = np.asfarray(ranking)[:k]
#     index_found_ranks = np.where(ranking.astype(int) == 1)[0]
#     if index_found_ranks.size > 0:
#         rank = index_found_ranks[0] + 1
#         return 1/float(rank)    
#     return 0

# def mean_reciprocal_rank(rankings, k=None):
#     """
#         Score is the mean of reciprocal ranks on an array of rankings. 
#         :param rankings: array of rankings, where ranking is a relevance score 
#             (list of numpy) in rank order (first element is the most relevant)
#         :type ranking: list, np.array
#         :return: mean reciprocal rank of a list of rankings 
#         :rtype: float
#     """

#     return np.mean([reciprocal_rank(ranking, k=k) for ranking in rankings])

# # def rank_accuracy(ranking):
# #   FUNCTION IN DEVELOPMENT
# #   ISSUE: when all correct items are in the ranking but all in the incorrect order, I want a 0.5 score
# #
# #     """
# #     Score is rank accuracy . Relevance is positive real values.  Can use binary
# #     as the previous methods.

# #     For a N-size list:
# #         If item is outside the sequence: 0 score
# #         Elif item is in sequence but in the wrong position: 1/N score
# #         Else (item in the sequence and in the right position): 1 score

# #     """    
# #     ranking = np.asfarray(ranking)
# #     ranking_ideal = np.asfarray(sorted(ranking, reverse=True))    
# #     ideal_index = np.argsort(-ranking_ideal)

# #     score = 0
# #     ranking_length = len(ranking)

# #     for item_index in np.arange(ranking_length):
# #         if ranking[item_index] == 0:
# #             item_score = 0
# #         elif item_index == ideal_index[item_index]:
# #             item_score = 1
# #         else:
# #             item_score = float(1/ranking_length)        
# #         score += item_score

# #     print ("Rank Acc for {}: {}".format(ranking, float(score/ranking_length)))
# #     return float(score/ranking_length)
    
# # def mean_rank_accuracy(rankings):
# #     """
# #         Score is the mean rank accuracy of a list of rankings. Relevance is positive real values.  Can use binary
# #         as the previous methods.
        
# #         :param rankings: a list of rankings
# #         :ptype: [list, np.array]
        
# #         :return: mean rank accuracy
# #         :rtype: float
# #     """
# #     return np.mean([rank_accuracy(ranking) for ranking in rankings])


# if __name__ == "__main__":
    
#     rankings = [[1, 1, 1, 1, 1, 1], # Totally right ranking
#                 [0, 0, 0, 0, 0, 0], # Totally wrong ranking
#                 [0, 0, 1, 0, 0, 1]] # Partially right ranking
#     k = 2

#     print ("-"*10)
#     print ("Precision@{}: {}".format(k, precision_at_k(rankings[-1], k)))

#     print ("-"*10)
#     print ("Reciprocal Rank: ", reciprocal_rank(rankings[-1]))
#     print ("Reciprocal Rank@{}: {}".format(k, reciprocal_rank(rankings[-1], k)))
#     print ("Mean Reciprocal Rank: ", mean_reciprocal_rank(rankings))
#     print ("Mean Reciprocal Rank@{}: {}".format(k, mean_reciprocal_rank(rankings, k)))


# import numpy as np
# # rs = np.array([1,2,3,4,5], [6,7,8,9,0])
# # rs = (np.asarray(r).nonzero()[0] for r in rs)
# # print(np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs]))

# l1 = [2,1,4,6,3,4]
# l2 = [12,41,23,2,3,10]
# l3 = [14,25,32,12,3,6]
# l = [l1, l2, l3]

# rs = (np.asarray(r).nonzero()[0] for r in l)
# print(np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs]))
# import numpy as np 

# def rr(ss):
#     i = 1
#     for s in ss:
#         print(s)
#         if s == True:
#             return 1.0 / float(i)

#         else:
#             i = i + 1

        
# def mrr(scores):
#     i = 1
#     result = 0
#     for score in scores:
#         result  = result + rr(score)
#         i = i + 1
#         # print(i)
#     return result / i
        
# if __name__ == '__main__':

#     m = np.array([[1,1,1,1,1],
#                   [0,0,0,0,0]])
#     print (mrr(m))
    global movie_imdb_rating
    movieIds1 = list(map(lambda pair: pair[0], ranking1))
    movieIds2 = list(map(lambda pair: pair[0], ranking2))
    m1 = []
    m2 = []
    for movieId in movieIds1:
        if movieIds1.index(movieId) != 0:
            m1.append(movieIds1.index(movieId))
        if movieIds2.index(movieId) != 0:
            m2.append(movieIds2.index(movieId))
    m = [m1, m2]
    mr = np.asarray(m)
    ss = mrr_score(mr)

    return ss

    global movie_imdb_rating
    movieIds1 = list(map(lambda pair: pair[0], ranking1))
    movieIds2 = list(map(lambda pair: pair[0], ranking2))
    distance = 0
    rank1 = 0
    for movieId in movieIds1:
        rank2 = movieIds2.index(movieId)
        distance += abs(rank1 - rank2)
        rank1 += 1
    return float(distance) / float(len(ranking1))

total = 0.0
l = [1,2,3,4,5]
n = 0
for i in l:
    n += 1
    total = 1/i + total
f = float(total/n)
print (f)