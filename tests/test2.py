from pathos.multiprocessing import ProcessingPool

from multiprocessing import Pool
import os

if __name__ == "__main__":
    print(os.getpid())
    print(os.cpu_count())

    with ProcessingPool(processes=4) as p:
        # results = []
        # for i in range(4):
        #     results.append(p.apply_async(os.getpid))
        # print([x.get(timeout=None) for x in results])
        multiple_results = [p.apipe(lambda x: x, ()) for i in range(4)]
        print([res.get(timeout=None) for res in multiple_results])
        # multiple_results = p.amap(lambda x: os.getpid(), range(10)).get()
        # print(multiple_results)