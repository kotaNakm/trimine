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
import math

TR=0.2

def factors_plot(trimine):    
    O, A, C = trimine.get_factors()
    outputdir = trimine.outputdir

    sns.set()
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



if __name__ == '__main__':

    input_tag = 'us_ele' #
    # input_tag = 'online_retail_a1' #(4631, 36, 17713)
    # input_tag = 'online_retail_a2' #(36, 4631, 17713)


    tensor = np.load(f'../{input_tag}.npy')
    outputdir = '../tirmine_result_stream/' + input_tag +'/'
    
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    
    u, v, n = tensor.shape
    k = 2

    #train
    train_n = math.floor(n*TR)

    print(f'n:{n}\n train_n:{train_n}')

    trimine = TriMine(k, u, v, train_n, outputdir)
    # Infer TriMine's parameters
    outputdir_s= outputdir+'train/'
    trimine.outputdir = outputdir_s
    if os.path.exists(outputdir_s):
        shutil.rmtree(outputdir_s)
    os.makedirs(outputdir_s)

    start_time = time.process_time()
    trimine.infer(tensor[:,:,:train_n], n_iter=20)#20 #50
    elapsed_time = time.process_time() - start_time
    print(f'Elapsed time(train): {elapsed_time:.2f} [sec]')
    trimine.save_model()
    factors_plot(trimine)



    #strem
    for i in range(train_n,n,100):
        
        outputdir_s=outputdir+str(i)+'/'
        trimine.outputdir = outputdir_s
        if os.path.exists(outputdir_s):
            shutil.rmtree(outputdir_s)
        os.makedirs(outputdir_s)
        
        start_time = time.process_time()
        trimine.infer_online(tensor[:,:,:i], n_iter=5,verbose=True)#20 #50
        elapsed_time = time.process_time() - start_time
        print(f'Elapsed time(online#{i}): {elapsed_time:.2f} [sec]')
        trimine.save_model()
        factors_plot(trimine)
        
