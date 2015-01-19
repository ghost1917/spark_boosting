
from collections import defaultdict
import time

def eval (dataset, classification_results):
    expected_targets = map (lambda x: x['t'], dataset)

    comparition = zip (expected_targets, classification_results)

    statistics = defaultdict (int)

    for element in comparition:
        statistics [element] += 1


    return {
            "true positive" : statistics [(1,1)],
            "false positive" : statistics [(0,1)],
            "true negative" : statistics [(0,0)],
            "false negative" : statistics [(1,0)],
            "precision": 1.*statistics [(1,1)] / (statistics [(1,1)]+statistics [(0,1)]),
            "recall": 1.*statistics [(1,1)] / (statistics [(1,1)]+statistics [(1,0)]),
            }



def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts)
        return result


    return timed
