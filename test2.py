""" Conduct experiments for TriMine """

import argparse
import os
import shutil
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trimine import TriMine


if __name__ == '__main__':

    # input_tag = 'us_ele' #
    # input_tag = 'online_retail_a1' #(4631, 36, 17713)
    input_tag = 'online_retail_a2' #(36, 4631, 17713)


    tensor = np.load(f'../{input_tag}.npy')
    outputdir = '../tirmine_result/' + input_tag +'/'
    
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    sns.set()

    u, v, n = tensor.shape
    k = 5
    trimine = TriMine(k, u, v, n, outputdir)

    # Infer TriMine's parameters
    start_time = time.process_time()
    trimine.infer(tensor, n_iter=50)#20
    elapsed_time = time.process_time() - start_time
    print(f'Elapsed time: {elapsed_time:.2f} [sec]')

    trimine.save_model()

    O, A, C = trimine.get_factors()

    plt.figure(figsize=(15,4))
    plt.plot(O)
    plt.title('Object matrix, O')
    plt.xlabel('Objects')
    plt.ylabel('Topic')
    plt.savefig(outputdir + 'O.png')
    plt.close()

    plt.figure(figsize=(15,4))
    plt.plot(A)
    plt.title('Actor matrix, A')
    plt.xlabel('Actors')
    plt.ylabel('Topic')
    plt.savefig(outputdir + 'A.png')
    plt.close()

    plt.figure(figsize=(15,4))
    plt.plot(C)
    plt.title('Time matrix, C')
    plt.xlabel('Time')
    plt.ylabel('Topic')
    plt.savefig(outputdir + 'C.png')
    plt.close()
