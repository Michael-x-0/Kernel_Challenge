from random_walk import RandonWalk
import pickle
import argparse

parser = argparse.ArgumentParser(description="build a part of kernel")
parser.add_argument('-i', default=1, type=int, help='the index of the part')
parser.add_argument('-m', default=1, type=int, help='the total number of part')
parser.add_argument('-r',  action='store_true', help='resume the fitting at r')
parser.add_argument('-k',  type=str, default='kernel', help='kernel name')
parser.add_argument('-mt',  type=int,default=-1, help='max step')

if __name__ == "__main__":
    with open("data-challenge-kernel-methods-2022-2023/training_data.pkl","rb") as train_data, open("data-challenge-kernel-methods-2022-2023/test_data.pkl","rb") \
        as test_data, open("data-challenge-kernel-methods-2022-2023/training_labels.pkl","rb") as train_label:
        train = pickle.load(train_data)
        test = pickle.load(test_data)
        label = pickle.load(train_label)
    
    args = parser.parse_args()
    whole = train + test
    walk = RandonWalk(dic=False)
    r=-1
    mt = args.mt
    kernel = args.k
    Kr = []
    if args.r:
        with open(f"{kernel}_{args.i}_chpt.pkl", "rb") as f:
            x = pickle.load(f)
            r = x['step']
            Kr = x['K']
    walk.fit_transform(whole,n=50,lamb=None,i=args.i,m= args.m,resume=r,max_step=mt, kernel=kernel,Kr = Kr)
