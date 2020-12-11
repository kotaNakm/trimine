""" Conduct experiments for TriMine """

import argparse
import os
import shutil
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trimine_HMM import TriMine
import math
import dill



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

    # input_tag = 'us_ele' #
    # input_tag = 'online_retail_a1' #(4631, 36, 17713)
    # input_tag = 'online_retail_a2' #(36, 4631, 17713)
    # input_tag = 'HVFTV_h_1';TR=0.5;width=20
    input_tag = 'HVFTV_h_1';TR=0.1;width=10
    # input_tag = 'HVFTV_m_1';TR=0.3;width=200

    tensor = np.load(f'../{input_tag}.npy')
    outputdir = '../trimine_result_stream_update/' + input_tag +'/'
    
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    
    u, v, n = tensor.shape
    k = 3

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
    tensor_T = tensor[:,:,:train_n]
    trimine.init_infer(tensor_T, n_iter=10)#10 #20 #50
    trimine.init_regime(tensor_T,0)

    elapsed_time = time.process_time() - start_time
    print(f'Elapsed time(train): {elapsed_time:.2f} [sec]')
    trimine.save_model()
    factors_plot(trimine)

    #stream
    start_time_stream = time.process_time()
    path = []
    path.append([0,0])
    times = []
    for i in range(train_n,n,width):
        outputdir_s=outputdir+str(i)+'/'
        trimine.outputdir = outputdir_s
        if os.path.exists(outputdir_s):
            shutil.rmtree(outputdir_s)
        os.makedirs(outputdir_s)
        
        start_time = time.process_time()
        shift_id = trimine.infer_online_HMM(tensor[:,:,i:i+width],n_iter=10,verbose=False)#20 #50
        elapsed_time = time.process_time() - start_time
        print(f'Elapsed time(online#{i}): {elapsed_time:.2f} [sec]')
        times.append(elapsed_time)
        trimine.save_model()
        factors_plot(trimine)
        if shift_id:
            prev_n = i
            path.append([shift_id,prev_n])

    elapsed_time = time.process_time() - start_time_stream
    print(f'Elapsed time(all stream processing): {elapsed_time:.2f} [sec]')
    result = [path,times,trimine]
    if True:
        dill.dump(result, open(f'{outputdir}result.dill','wb'))
