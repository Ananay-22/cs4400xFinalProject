import recordlinkage as rl;
from os.path import join;
from recordlinkage.base import BaseCompareFeature;
import pandas as pd;
import spacy;
import Levenshtein as lev;
from functools import lru_cache
import sys
from time import time

_nlp = spacy.load("en_core_web_sm")

# for progress bar loading
total = -1
start_time = 0



@lru_cache(3000)
def nlp(s):
    return _nlp(s)


def jaccard_similarity(x, y):
    x = set(x.lower().split())
    y = set(y.lower().split())
    return len(x.intersection(y)) / max(len(x), len(y))

def levenshtein_distance(x, y):
    x = x.lower()
    y = y.lower()
    return lev.distance(x, y)

class NLP_POS_Comparator(BaseCompareFeature):
    def __inner_compute_vectorized(self, s1, s2):
        self.counter += 1
        sys.stdout.write("\rCompleted %3.2f%% [%8s/%8s] in %d seconds." % (self.counter/total*100, self.counter, total, time() - start_time))
        sys.stdout.flush()
        try:
            document1 = nlp(s1)
            document2 = nlp(s2)
            final_score = 0
            for ent1 in set(document1.ents):
                for ent2 in set(document2.ents):
                    score = max((int(0.32 * len(ent1.text + ent2.text)) - levenshtein_distance(ent1.text, ent2.text)) / int(
                        0.32 * len(ent1.text + ent2.text)), 0) + jaccard_similarity(ent1.text, ent2.text)
                    if score > 0.65 and ent1.label_ == ent2.label__:
                        final_score+=3.0
                    final_score += score
            for token1 in document1:
                for token2 in document2:
                    score = max((int(0.32*len(token1.text+token2.text))-levenshtein_distance(token1.text, token2.text))/int(0.32*len(token1.text+token2.text)), 0) + jaccard_similarity(token1.text, token2.text)
                    if score > 0.9 and token1.pos_ == token2.pos_:
                        final_score += 2.0
                    final_score += score
            return final_score
        except:
            return 0

    def _compute_vectorized(self, s1, s2):
        self.counter = 0
        return pd.DataFrame(data = {'l': s1, 'r': s2}).apply(lambda p: self.__inner_compute_vectorized(p['l'], p['r']), axis=1)


def main():
    # load in dataset
    global total, start_time
    ltable = pd.read_csv(join('data', "ltable.csv"));
    rtable = pd.read_csv(join('data', "rtable.csv"));

    indexer = rl.Index();
    indexer.sortedneighbourhood(left_on='brand', right_on='brand');
    blocked = indexer.index(ltable, rtable);
    total  = len(blocked)
    print("Total number of tuples to process: %s tuples" % total);
    compare = rl.Compare();

    compare.add(NLP_POS_Comparator('title', 'title', label='custom'))
    start_time = time()
    features = compare.compute(blocked, ltable, rtable);
    print()

    features.sum(axis=1).value_counts().sort_index(ascending=False);

    potential_matches = features.reset_index()
    potential_matches = pd.merge(potential_matches, ltable, how="left", left_on="level_0", right_on='id');
    potential_matches = pd.merge(potential_matches, rtable, how="left", left_on="level_1", right_on='id');

    train = pd.read_csv('data/train.csv')
    potential_matches.merge(train, how="outer", left_on=["level_0", "level_1"], right_on=["ltable_id", "rtable_id"], indicator=True).query("_merge=='left_only'")
    potential_matches[potential_matches['custom'] > 20][['level_0', 'level_1', 'custom']].sort_values(by=['custom'],
                                                                                                      ascending=False)[
        ['level_0', 'level_1']].rename(
        {'level_0': 'ltable_id', 'level_1': 'rtable_id'}, inplace = False).head(
        650).to_csv('data/output.csv', index=False)

if __name__ == '__main__':
    main();
