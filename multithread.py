import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

from tqdm.contrib.concurrent import process_map  # or thread_map
import time



def f(a,b):
  print(f"({a},{b})")

def multi():
     with Pool(processes=multiprocessing.cpu_count()) as pool:
        process_map(pool.map(partial(f,2),enumerate([1,2,3,4])),max_workers=12)


if __name__ =="__main__":
    #multi()
    pbar = tqdm(total=100)
    for i in range(10):
        time.sleep(5)
        pbar.update(10)
    pbar.close()
