""" Python implementation of TriMine @ KDD'12 """

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, loggamma
from tqdm import trange
import numba
import copy
from hmmlearn.hmm import GaussianHMM
import time
import statistics

#for omiting warnings
# from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore')

MINK = 1
MAXK = 5
N_INFER_ITER_HMM = 5
ZERO = 1.e-8
FB = 4 * 8
TB = 10 #0.1 1 #10  # 100 #1000 #transiton bias

# https://github.com/scikit-learn/scikit-learn/blob/7e85a6d1f/sklearn/decomposition/_online_lda.py#L135


class Regime(object):
    def __init__(self):
        self.costM = np.inf
        self.costC = np.inf
        self.costT = np.inf
        
        """
        k,u,v,n
        O,A,C
        alpha,beta
        model
        """

    def get_params(self, **kwargs):
        return self.alpha, self.beta, self.gamma

    def get_factors(self):
        return self.O, self.A, self.C

    def compute_costM(self):
        cost=0
        k = self.model.n_components
        print(k) 
        print(costHMM(k,self.k))       
        cost += costHMM(k,self.k)
        cost += FB * 3
        self.costM = cost
        print(cost)
        return cost

    def compute_costC(self,seq,pre_n,n):
        #Gaussian HMM returns log prob
        
        # llh = self.model.score(seq)#/(n - pre_n)
        # cost = -llh / np.log(2)
        # self.costC = cost*(n - pre_n)
        cost = - self.model.score(seq) * TB
        print(f'costC:{cost}')
        return cost

    # def compute_total(self):
    #     llh = loggamma(self.alpha * self.k) - self.k * loggamma(self.alpha)
    #     # llh += loggamma(self.alpha * self.u) - self.u * loggamma(self.alpha)
    #     llh += loggamma(self.beta * self.k) - self.k * loggamma(self.beta)
    #     print(f'llh:{llh}')
    #     for i in range(self.k):
    #         llh += (self.alpha - 1) * sum([np.log(self.O[j, i]) for j in range(self.u)]) / self.u
    #         llh += (self.beta  - 1) * sum([np.log(self.A[j, i]) for j in range(self.v)]) / self.v
    #     print(f'llh:{llh}')
    #     costC_hyp = - llh
    #     costC_hmm = - self.model.score(self.C)/ np.log(2)
    #     self.costC =  costC_hmm + costC_hyp
    #     print(costC_hyp)
    #     print(costC_hmm)
    #     print(self.costC)


class TriMine(object):
    def __init__(self, k, u, v, n, outputdir):
        # statuses
        self.k = k  # of topics
        self.u = u  # of objects
        self.v = v  # of actors
        self.n = n  # data duration
        self.outputdir = outputdir
        self.train_log = []
        self.max_alpha = 100 #0.001
        self.max_beta  = 1 #100
        self.max_gamma = 1 #100
        self.init_params()

        self.prev_dist_O = np.full((self.u),1)
        self.prev_dist_A = np.full((self.k),1)
        self.prev_dist_C = np.full((self.k),1)

        self.regimes=[]
        r_g = Regime()
        self.regimes.append(r_g)

    def init_params(self, alpha=None, beta=None, gamma=None):
        """ Initialize model parameters """
        # if parameter > 1: pure
        # if parameter < 1: mixed
        self.alpha = np.full(self.u,50/self.k) #0.0005/self.k #  #self.u
        self.beta  = np.full(self.k,0.1)#5  #self.v
        self.gamma = np.full(self.k,0.1)#5 #self.n
        self.O = np.zeros((self.u, self.k))  # Object matrix
        self.A = np.zeros((self.v, self.k))  # Actor matrix
        self.C = np.zeros((self.n, self.k))  # Time matrix   

    def init_status(self,tensor):
        self.Nk = np.zeros(self.k, dtype=int)
        self.Nu = np.zeros(self.u, dtype=int)
        self.Nku = np.zeros((self.k, self.u), dtype=int)
        self.Nkv = np.zeros((self.k, self.v), dtype=int)
        self.Nkn = np.zeros((self.k, self.n), dtype=int)
        # self.Z = np.full((self.u, self.v, self.n), -1)
        self.Nsum = tensor.sum()
        self.Z = np.full((self.Nsum), -1)

        # self.prev_Nk = np.zeros(self.k, dtype=int)
    
    def Save_prev_status(self):
        self.prev_Nk = copy.deepcopy(self.Nk)
        self.prev_Nu = copy.deepcopy(self.Nu)
        self.prev_Nku = copy.deepcopy(self.Nku)
        self.prev_Nkv = copy.deepcopy(self.Nkv)
        self.prev_Nkn = copy.deepcopy(self.Nkn)
        self.prev_Nsum = copy.deepcopy(self.Nsum)
        self.prev_Z = copy.deepcopy(self.Z)

    def Undo_prev_status(self):
        self.Nk = copy.deepcopy(self.prev_Nk)
        self.Nu = copy.deepcopy(self.prev_Nu)
        self.Nku = copy.deepcopy(self.prev_Nku)
        self.Nkv = copy.deepcopy(self.prev_Nkv)
        self.Nkn = copy.deepcopy(self.prev_Nkn)
        self.Nsum =copy.deepcopy( self.prev_Nsum)
        self.Z = copy.deepcopy(self.prev_Z)
        print('undo prev rgm')

    def update_status(self,tensor,prev_n):

        

        #init param regime変わったらtrackしているparameterを更新前にもどしたい
        # prev_rgm = self.regimes[self.prev_rgm_id]
        # self.alpha = prev_rgm.alpha
        # self.beta  = prev_rgm.beta
        # self.gamma = prev_rgm.gamma
        print(f'alpha:{self.alpha}')
        print(f'beta:{self.beta}')
        print(f'gamma{self.gamma}')


    def get_params(self, **kwargs):
        return self.alpha, self.beta, self.gamma

    def get_factors(self):
        return self.O, self.A, self.C

    def init_regime(self,tensor,rgm_id):
        regime = self.regimes[rgm_id]

        # self.update_O_A(tensor,regime.model)
        regime.k = self.k;regime.u = self.u;regime.v = self.v;regime.n = self.n
        regime.alpha = self.alpha;regime.beta = self.beta;regime.gamma = self.gamma    
        regime.O = self.O;regime.A = self.A;regime.C = self.C

        regime.model = self.model
        regime.costM = regime.compute_costM()

        print(regime.costC)
        print(regime.costM)
        self.prev_rgm_id = 0

        # self.regimes.append(regime)
        # 
        # print(len(self.regimes))


    def init_infer(self, tensor, n_iter=10, tol=1.e-8, init=True, verbose=True):
        """
        Given: a tensor (actors * objects * time)
        Find: matrices, O, A, C
        """

        if init == True:
            self.init_status(tensor)
        
        self.Nu = np.sum(tensor,axis=(1,2))
        for iteration in range(n_iter):
            # Sampling hidden topics z, i.e., Equation (1)
            self.Z = self.sample_topic(tensor, self.Z, 0,0)
        
            # self.update_alpha()
            # self.update_beta()
            # self.update_gamma(0)
            
            # Compute log-likelihood
            llh = 0#self.loglikelihood(0,self.alpha,self.beta,self.gamma,cnt)
            self.train_log.append(llh)

            # Early break
            # if iteration > 0:
            #     if np.abs(self.train_log[-1] - self.train_log[-2]) < tol:
            #         print('Early stopping')
            #         break
        
        self.compute_factors(0)
        self.update_prev_dist()
        self.model = self.estimate_hmm(ZnormSequence(self.C))
        self.Save_prev_status() 
        self.init_regime(tensor,0)
        self.regimes[0].costC = llh 
        self.all_C = copy.deepcopy(self.C)
        self.cur_n = copy.deepcopy(self.n)
        self.vscost_log=[]

        if verbose == True:
            # Print learning log
            print('Iteration', iteration + 1)
            print('loglikelihood=\t', llh)
            print(f'| alpha\t| {self.alpha}')
            print(f'| beta \t| {self.beta} ')
            print(f'| gamma\t| {self.gamma}')
            # Save learning log
            plt.plot(self.train_log)
            plt.xlabel('Iterations')
            plt.ylabel('Log-likelihood')
            plt.savefig(self.outputdir + 'train_log.png')
            plt.close()        

    def infer_online_HMM(self, tensor, n_iter=10, tol=1.e-8,verbose=True):
        
        """
        Given: a tensor (actors * objects * time)
        Find: matrices, O, A, C
        """ 
        
        if not tensor.ndim == 3 :
            tensor = tensor[:,:,np.newaxis] 

        u,v,n = tensor.shape
        self.init_status(tensor)
        if True:
            self.init_params()
        self.n = n
        self.cur_n = self.n + self.cur_n
        #update C mat
        self.C = np.zeros((self.n, self.k)) 

        # self.update_status(tensor,pre_n)
        self.train_log = []

        self.Nu = np.sum(tensor,axis=(1,2))    

        # print(self.Nk)
        # print(self.Nu)
        # print(self.Nku)
        # print(self.Nkv)
        # print(self.Nkn)
        # print(self.O)

        for iteration in range(n_iter):
            # Sampling hidden topics z, i.e., Equation (1)
            self.Z = self.sampleZ_pickC(tensor, self.model, self.Z)

            # Update parameters
            self.update_alpha()
            self.update_beta()
            self.update_gamma(0)
            
            # Compute log-likelihood
            llh = 0#self.loglikelihood(pre_n,self.alpha,self.beta,self.gamma,new_cnt-cnt)
            self.train_log.append(llh)

            # Early break
            # if iteration > 0:
            #     if np.abs(self.train_log[-1] - self.train_log[-2]) < tol:
            #         print('Early stopping')
            #         break
            #         self.compute_factors(pre_n,cur_n)
        
            if verbose == True:
                # Print learning log
                # print('Iteration', iteration + 1)
                print('loglikelihood=\t',llh)
                print(f'| alpha\t| {self.alpha}')
                print(f'| beta \t| {self.beta} ')
                print(f'| gamma\t| {self.gamma}')
                # Save learning log
                plt.plot(self.train_log)
                plt.xlabel('Iterations')
                plt.ylabel('Log-likelihood')
                plt.savefig(self.outputdir + 'train_log.png')
                plt.close()

        self.compute_factors(1)
        self.update_prev_dist()
        # print(self.all_C.shape)
        # print(self.C.shape)

        self.all_C = np.concatenate([self.all_C,self.C],0)
        self.C = copy.deepcopy(self.all_C)
        shift_id = self.model_compressinon(self.cur_n-self.n)
         
        return shift_id

    def loglikelihood(self,pre_n,alpha,beta,gamma,cnt):
        """ Compute Log-likelihood """

        # Symmetric dirichlet distribution
        # https://en.wikipedia.org/wiki/Dirichlet_distribution

        # print(self.n)
        # print(pre_n)
        # print(range(pre_n,self.n))
        # print(self.C.shape)

        #version python
        # llh = 0
        # llh = loggamma(self.alpha * self.k) - self.k * loggamma(self.alpha)
        # # llh += loggamma(self.alpha * self.u) - self.u * loggamma(self.alpha)
        # llh += loggamma(self.beta * self.k) - self.k * loggamma(self.beta)
        # llh += loggamma(self.gamma * self.k) - self.k * loggamma(self.gamma)

        # for i in range(self.k):
        #     llh += (self.alpha - 1) * sum([np.log(self.O[j, i]) for j in range(self.u)]) / self.u
        #     llh += (self.beta  - 1) * sum([np.log(self.A[j, i]) for j in range(self.v)]) / self.v
        #     llh += (self.gamma - 1) * sum([np.log(self.C[j, i]) for j in range(pre_n,self.n)]) / (self.n-pre_n)

        # return llh

        cur_n = self.n
        n = self.n - pre_n
        #version C
        llh = 0
        llh = self.u * loggamma(alpha * self.k) - self.u * self.k * loggamma(alpha)
        llh += self.k * loggamma(beta * self.v) - self.k * self.v * loggamma(beta)
        # llh += self.k * loggamma(gamma * self.n) - self.k * self.n * loggamma(gamma)
        llh += self.k * loggamma(gamma * n) - self.k * n * loggamma(gamma)


        for i in range(self.u):
            llh -= loggamma(self.Nu[i] + alpha * self.k)
            for j in range(self.k):
                llh += loggamma(self.Nku[j][i] + alpha)
        for i in range(self.k):
            llh -= loggamma(self.Nk[i] + beta * self.v)
            llh -= loggamma(self.Nk[i] - self.prev_Nk[i] + gamma * n)
            # llh -= loggamma(self.Nk[i] + gamma * self.n)
            for j in range(self.v):
                llh += loggamma(self.Nkv[i][j] + beta)
            for j in range(pre_n,cur_n):
                llh += loggamma(self.Nkn[i][j] + gamma)
        
        # llh -= loggamma(cnt)
        # print(loggamma(cnt))

        # print(f'div{np.log(cnt * 1/alpha * 1/beta *1/gamma)}')
        # print(f'llh:{llh}')

        return llh/np.log(cnt)#* 1/alpha * 1/beta *1/gamma)

    def sample_topic(self, X, Z, pre_n,cnt):
        return _sample_topic(self.Nk,self.Nu,self.Nku,self.Nkv,self.Nkn,self.k,self.u,self.v,self.n,self.alpha,self.beta,self.gamma,X,Z,pre_n,cnt)

    def sampleZ_pickC(self, X, model, Z):
        return _sampleZ_pickC(self.Nk,self.Nu,self.Nku,self.Nkv,self.Nkn,self.prev_Nk,self.k,self.u,self.v,self.n,self.alpha,self.beta,self.gamma,self.prev_dist_O,self.prev_dist_A,self.prev_dist_C,X,model,Z)


    def update_alpha(self):
        # print(self.alpha)
        for l in range(self.u):
            if self.Nu[l] < 1:
                # print('continue')
                continue
            alpha = self.alpha[l]
            prev_dist = self.prev_dist_O[l]

            num = -1 * self.k * prev_dist * digamma(alpha*prev_dist)
            den = -1 * digamma(alpha)
            for i in range(self.k):
                num += prev_dist * digamma(self.Nku[i,l] + alpha * prev_dist)
            den += digamma(self.Nu[l] + alpha)
            
            # print(f'num::{num}')
            # print(f'den::{den}')
            alpha *= num / den

            if alpha > self.max_alpha:
                alpha = self.max_alpha
            elif alpha < 1.e-8:
                alpha = 1.e-8
            self.alpha[l] = alpha 
        # print(self.alpha)
    
    def update_beta(self):   
        # print(self.beta)
        for l in range(self.k):
            if self.Nu[l] < 1:
                # print('continue')
                continue

            beta = self.beta[l]
            prev_dist = self.prev_dist_A[l]
            # print(prev_dist)
            num = -1 * self.v * prev_dist * digamma(beta * prev_dist)
            # num = -1 *prev_dist * digamma(beta * prev_dist)
            den = -1 * digamma(beta)
            # print(f'num::{num}')
            # print(f'den::{den}')
            for i in range(self.v):
                num += prev_dist * digamma(self.Nkv[l,i] + beta * prev_dist)
            den += digamma(self.Nk[l] + beta)
            # den += digamma(self.Nkv[l,:].sum() + beta)
            # print(a,self.Nk[l])
            # print(np.sum(self.Nkv[l,:]),self.Nk[l])
            # print(f'num::{num}')
            # print(f'den::{den}')
            beta *= num / den

            if beta > self.max_beta:
                beta = self.max_beta
            elif beta < 1.e-8:
                beta = 1.e-8
            self.beta[l] = beta 
        # print(self.beta)
        # num = -1 * self.k * self.v * digamma(self.beta)
        # den = -1 * self.k * self.v * digamma(self.beta * self.v)

        # for i in range(self.k):
        #     den += self.v * digamma(self.Nk[i] + self.beta * self.v)
        #     for j in range(self.v):
        #         num += digamma(self.Nkv[i, j] + self.beta)

        # self.beta *= num / den

        # if self.beta > self.max_beta:
        #     self.beta = self.max_beta
        # elif self.beta < 1.e-8:
        #     self.beta = 1.e-8

    def update_gamma(self,pre_n):

        # print(self.gamma)
        for l in range(self.k):
            if self.Nk[l] < 1:
                # print('continue')
                continue
            gamma = self.gamma[l]
            prev_dist = self.prev_dist_C[l]
            # print(prev_dist)

            num = -1 * self.n * prev_dist * digamma(gamma* prev_dist)
            den = -1 * digamma(gamma)
            for i in range(self.n):
                num += prev_dist * digamma(self.Nkn[l,i] + gamma * prev_dist)
            den += digamma(self.Nk[l] + gamma)
            # den += digamma(self.Nkn[l,:].sum() + gamma)
                
            gamma *= num / den

            if gamma > self.max_gamma:
                gamma = self.max_gamma
            elif gamma < 1.e-8:
                gamma = 1.e-8
            self.gamma[l] = gamma
        # print(self.gamma)
        # print(pre_n)
        # print(self.n)
        # print(self.Nkn.shape)
        
        # print(self.n-pre_n)
        # print(self.Nk)
        # print(np.sum(self.Nk))
        # print(self.Nkn)

        # n = self.n - pre_n
        # num = -1 * self.k * n * digamma(self.gamma)
        # den = -1 * self.k * n * digamma(self.gamma * n)

        # for i in range(self.k):
        #     # den += n * digamma(self.Nk[i] + self.gamma * n)
        #     den += n * digamma(self.Nk[i] -self.prev_Nk[i] + self.gamma * n)
        #     for j in range(pre_n,self.n):
        #         num += digamma(self.Nkn[i, j] + self.gamma)

        # num = -1 * self.k * (self.n-pre_n) * digamma(self.gamma)
        # den = -1 * self.k * (self.n-pre_n) * digamma(self.gamma * (self.n-pre_n))
        # # print(num)
        # # print(den)
        # # print('============')
        # for i in range(self.k):
        #     den += self.n * digamma(self.Nk[i] + self.gamma * (self.n-pre_n))
        #     for j in range(pre_n,self.n):
        #         # print(j)
        #         num += digamma(self.Nkn[i, j] + self.gamma)
        #         print(num)
        # print(num)
        
        # self.gamma *= num / den

        # if self.gamma > self.max_gamma:
        #     self.gamma = self.max_gamma
        # elif self.gamma < 1.e-8:
        #     self.gamma = 1.e-8
    def update_prev_dist(self):

        self.prev_dist_O = np.sum(self.O,axis=1)/np.sum(self.O) 
        self.prev_dist_A = np.sum(self.A,axis=0)/np.sum(self.A)
        self.prev_dist_C = np.sum(self.C,axis=0)/np.sum(self.C)

        # print(self.prev_dist_O)
        # print(self.prev_dist_A)
        # print(self.prev_dist_C)

    def compute_factors(self,pre_n):
        """ 
        Generate three factors/matrices, O, A, and C
        """
        
        # self.O = np.zeros((self.u, self.k))
        # self.A = np.zeros((self.v, self.k))
        # self.C = np.zeros((self.n, self.k))

        # if pre_n:
        #     tmp_C = copy.deepcopy(self.C)
        #     self.C = np.zeros((self.n, self.k))
        #     self.C[:pre_n,:] = tmp_C[:pre_n,:]
        # self.alpha
        # beta = self.beta
        # gamma = self.gamma

        n = self.n
        for i in range(self.k):
            for j in range(self.u):
                self.O[j, i] = (
                    (self.Nku[i, j] +self.alpha[j] * self.prev_dist_O[j])
                    / (self.Nu[j] + self.alpha[j]))
            for j in range(self.v):
                self.A[j, i] = (
                    (self.Nkv[i, j] + self.beta[i] * self.prev_dist_A[i])
                    / (self.Nk[i] + self.beta[i]))   
            if pre_n:
                for j in range(self.n):
                    self.C[j, i] = (
                        (self.Nkn[i, j] + self.gamma[i] * self.prev_dist_C[i])
                        / (self.Nk[i] + self.gamma[i]))  * self.Nk.sum() / self.norm_Nk
            else:
                for j in range(self.n):
                    self.C[j, i] = (
                        (self.Nkn[i, j] + self.gamma[i] * self.prev_dist_C[i])
                        / (self.Nk[i] + self.gamma[i]))
                self.norm_Nk = self.Nk.sum()
        
        
        # for j in range(pre_n,cur_n):
        #     sum_c = self.C[j,:].sum()
        #     print(sum_c)
        #     self.C[j,:] = self.C[j,:] /sum_c

        # print(self.C[:,i])
                    
        # print(self.O.sum(axis=1))
        # print(self.A.sum(axis=0))
        # print(self.C.sum(axis=0))
        return self.O, self.A, self.C

    def estimate_hmm(self,X):
        min_ = np.inf
        opt_k = MINK
        # pp.pprint(regime.subs)
        for k in range(MINK, MAXK):
            prev = min_
            model = _estimate_hmm_k(X,k)
            score = model.score(X)
            # print(regime.costT)
            if score > prev:
                opt_k = k - 1
                break
        if opt_k < MINK: opt_k = MINK
        if opt_k > MAXK: opt_k = MAXK
        print(f'opt_k:{opt_k}')

        return _estimate_hmm_k(X, opt_k)
        # fit_predict = self.model.predict(X)
        # print(self.model.startprob_)
        # print(fit_predict)
        # print(self.model.get_stationary_distribution())
        # print(self.model.means_)
        # print(self.model.covars_)

    def model_compressinon(self,pre_n):
        shift_id = False

        cur_C = ZnormSequence(self.C)[pre_n:,:]

        prev_rgm = self.regimes[self.prev_rgm_id]
        candidate_rgm = Regime()
        candidate_rgm.model = self.estimate_hmm(cur_C)
        candidate_rgm.k = self.k
        candidate_rgm.alpha = copy.deepcopy(self.alpha);candidate_rgm.beta = copy.deepcopy(self.beta);candidate_rgm.gamma = copy.deepcopy(self.gamma)
        candidate_rgm.O = copy.deepcopy(self.O);candidate_rgm.A = copy.deepcopy(self.A)

        ## compute_costM 
        costM = candidate_rgm.compute_costM()#/(self.n - pre_n)
        costC = candidate_rgm.compute_costC(cur_C,0,0)
        cost_1 = costC + costM
        # cost_1 =  costC / (self.n - pre_n) + costM / new_cnt - self.prev_cnt

        print(f'new_regime:::{cost_1}')
        
        #直近とコスト比較
        # min_ = - self.loglikelihood(pre_n,prev_rgm.alpha,prev_rgm.beta,prev_rgm.gamma,new_cnt-self.prev_cnt)
        cost_0 = prev_rgm.compute_costC(cur_C,0,0)
        print(f'prev_regime:::{cost_0}')

        self.vscost_log.append([cost_0,cost_1,cost_1-cost_0])

        print('=========================================')
        print(f"{cost_0} vs {cost_1}")
        print('=========================================')
        print(f'diff::{cost_1-cost_0}')

        if cost_0 < cost_1: #stay
            print('stay')

        else: #shift to any regime
            shift_id = len(self.regimes) #index + 1
            min_ = cost_1
            add_flag = True

            #regime comparison
            for rgm_id,rgm  in enumerate(self.regimes):
                if rgm_id == self.prev_rgm_id:
                    continue
                else:
                    cost_0 = rgm.compute_costC(cur_C,0,0)
                    if cost_0 < min_:
                        shift_id = rgm_id
                        add_flag = False
                        min_ = cost_0

            print(f'shift::{pre_n}')
            print(f'{self.prev_rgm_id}===>>>{shift_id}')

            if add_flag: #add candidate  regime
                
                self.regimes.append(candidate_rgm)
                self.prev_rgm_id = shift_id
                
            else: # use existed regime
                shift_rgm = self.regimes[shift_id]
                self.alpha  = copy.deepcopy(shift_rgm.alpha)
                self.beta =copy.deepcopy(shift_rgm.beta)
                self.gamma = copy.deepcopy(shift_rgm.gamma)
                self.prev_rgm_id = shift_id

        return shift_id #shift先のregime番号

    def rgm_update_fin(self):
        if self.prev_rgm_id == (len(self.regimes)-1):
            rgm = self.regimes[self.prev_rgm_id]
            rgm.alpha  = copy.deepcopy(self.alpha)
            rgm.beta = copy.deepcopy(self.beta)
            rgm.gamma = copy.deepcopy(self.gamma)
            rgm.O = copy.deepcopy(self.O)
            rgm.C = copy.deepcopy(self.C)

    def save_model(self):
        """ Save all of parameters for TriMine
        """
        with open(self.outputdir + 'params.txt', 'a') as f:
            f.write(f'topic,{self.k}\n')
            f.write(f'alpha,{self.alpha}\n')
            f.write(f'beta, {self.beta}\n')
            f.write(f'gamma,{self.gamma}\n')
            for costs in self.vscost_log:
                f.write(f'{costs}\n')

        np.savetxt(self.outputdir + 'O.txt', self.O)
        np.savetxt(self.outputdir + 'A.txt', self.A)
        np.savetxt(self.outputdir + 'C.txt', self.C)
        np.savetxt(self.outputdir + 'train_log.txt', self.train_log)

@numba.jit #(nopython=True)
def _sample_topic(Nk,Nu,Nku,Nkv,Nkn,k,u,v,n,alpha,beta,gamma,X,Z,pre_n,prev_cnt):
    """
    X: event tensor
    Z: topic assignments of the previous iteration
    """
  
    cnt = 0
    # for t in trange(self.n, desc='#### Infering Z'):
    # cnt=cnt
    for t in range(n):
        for i in range(u):
            for j in range(v):
                # for each non-zero event entry,
                # assign latent topic, z
                count =  X[i, j, t]
                for _ in range(count):
                    topic = Z[cnt]
                    if count == 0:
                        continue
                    if not topic == -1:
                        Nk[topic] -= 1
                        Nku[topic, i] -= 1
                        Nkv[topic, j] -= 1
                        Nkn[topic, t] -= 1
                        #↑ここで0になることがある(count loopなければ0以下もありうる)
                        #そしたらトピックを-1にする??
                        # if ((Nk  < 0).sum() > 0 or
                        #     (Nkv < 0).sum() > 0 or
                        #     (Nku < 0).sum() > 0 or
                        #     (Nkn < 0).sum() > 0):
                        #     print("Invalid counter N has been found")
                        #     # print(self.Nk,self.Nkv,self.Nku,self.Nkn)

                    """ compute posterior distribution """
                    posts = np.zeros(k)
                    # print(self.Nku[:, i])
                    # print(self.Nkv[:, j])
                    # print(self.Nkn[:, t])
                    for r in range(k):
                        # NOTE: Nk[r] = Nkv[r, :].sum() = Nkn[r, :].sum()
                        O = A = C = 1
                        O = (Nku[r, i] + alpha[i]) / (Nu[i] + alpha[i] * k)
                        A = (Nkv[r, j] + beta[r]) / (Nk[r] + beta[r]  * v)
                        C = (Nkn[r, t] + gamma[r]) / (Nk[r] + gamma[r] * n)
                        # print(O.shape,A.shape,C.shape)
                        # posts[r] = float('{:.5g}'.format(O * A * C))
                        posts[r] = O * A * C
                    posts = posts /posts.sum() # normalize with bias
                    # if posts.sum()>1 or posts.sum()<0:
                    #     print(posts)
                    #to avoid greater than one
                    #https://github.com/numba/numba/issues/3426

                    topic = np.argmax(np.random.multinomial(1, posts))
                    # print(topic, '<-', posts)
                    Z[cnt] = topic
                    Nk[topic] += 1
                    Nku[topic, i] += 1
                    Nkv[topic, j] += 1
                    Nkn[topic, t] += 1
                    cnt+=1
    return Z

@numba.jit #(nopython=True)
def _sampleZ_pickC(Nk,Nu,Nku,Nkv,Nkn,prev_Nk,k,u,v,n,alpha,beta,gamma,prev_dist_O,prev_dist_A,prev_dist_C,X,model,Z):

    # C_sample = model.sample(n-pre_n)[0]
    # C_sample = np.where(C_sample<ZERO,ZERO,C_sample)
    # print(C_sample)
    # print(pre_n,n)
    
    # means=[2,2,2,2]
    # sum_means = 8


    cnt = 0
    # states = model.sample(n-pre_n)[1]
    # means_all = model.means_
    # means_all = np.where(means_all < ZERO,ZERO,means_all)
    for t in range(n):
        # state = states[t-pre_n]
        # means = means_all[state]
        # sum_means = np.sum(means)

        #case:毎回sampling
        # state = model.sample(t-pre_n)[1][-1]
        # means = model.means_[state]
        # means = np.where(means < ZERO,ZERO,means)
        # sum_means = np.sum(means)

        for i in range(u):
            for j in range(v):
                # for each non-zero event entry,
                # assign latent topic, z
                count =  X[i, j, t]
                if count == 0:
                        continue
                # tmp_C = np.zeros((count,k))
                for tmp_cnt in range(count):
                    topic = Z[cnt]
                    if not topic == -1:
                        Nk[topic] -= 1
                        Nku[topic, i] -= 1
                        Nkv[topic, j] -= 1
                        Nkn[topic, t] -= 1

                    """ compute posterior distribution """
                    posts = np.zeros(k)
                    # print(self.Nku[:, i])
                    # print(self.Nkv[:, j])
                    # print(self.Nkn[:, t])
                    for r in range(k):
                        # NOTE: Nk[r] = Nkv[r, :].sum() = Nkn[r, :].sum()
                        O = A = C = 1
                        O = (Nku[r, i] + alpha[i] * prev_dist_O[i]) / (Nu[i] + alpha[i])
                        A = (Nkv[r, j] + beta[r] * prev_dist_A[r]) / (Nk[r] + beta[r])
                        C = (Nkn[r, t] + gamma[r]* prev_dist_C[r]) / (Nk[r] + gamma[r])
                        # print(O,A,C)
                        # C = (Nkn[r, t] + gamma + means[r]) / (Nk[r] + gamma * n + sum_means)
                        # C = C_sample[r,t]
                        # C = model.sample(n-pre_n)[0][r,t] 
                        # C = ZERO if C < ZERO else C
                        # tmp_C[tmp_cnt,r] = C
                        
                        # print(O.shape,A.shape,C.shape)
                        # posts[r] = float('{:.5g}'.format(O * A * C))
                    #     print(Nku[r, i],alpha ,Nu[i])
                    #     print(O,A,C)
                    #     old = (Nkn[r, t]+gamma)/(Nk[r] + gamma * n)
                        posts[r] = O * A * C
                        # if posts[r]>1 or posts[r]<0:
                        #     print(O,A,C)
                        #     print(Nk)
                        #     print(prev_Nk)
                    #     print(O * A * old)
                    #     print(posts[r])
                    # print('==========')
                    
                    posts = posts /(posts.sum()) # normalize with bias
                    # if posts.sum()>1 or posts.sum()<0:
                    #     print(posts)
                    #to avoid greater than one
                    #https://github.com/numba/numba/issues/3426

                    topic = np.argmax(np.random.multinomial(1, posts))
                    # print(topic, '<-', posts)
                    Z[cnt] = topic
                    Nk[topic] += 1
                    Nku[topic, i] += 1
                    Nkv[topic, j] += 1
                    Nkn[topic, t] += 1

                    cnt+=1

                # print(tmp_C)
                # print(tmp_C.shape)
                # print(np.mean(tmp_C,axis=0))
                # mat_C[t,:] = np.mean(tmp_C,axis=0)
    return Z


def _estimate_hmm_k(X,k=1):
    model = GaussianHMM(n_components=k,
                               covariance_type='diag',
                               n_iter=N_INFER_ITER_HMM)
    model.fit(X)
    return model
    
def log_s(x):
    return 2. * np.log2(x) + 1.

def costHMM(k, d):
    return FB * (k + k ** 2 + 2 * k * d) + 2. * np.log(k) / np.log(2.) + 1.


def ZnormSequence(seq,MAXVAL=1):
    #expect shape(length,dim)
    n_dims=seq.shape[1]
    normSeq = np.zeros(seq.shape)

    for i,seq_d in enumerate(seq.T):
        std = statistics.pstdev(seq_d)
        mean = statistics.mean(seq_d)

        normSeq[:,i]=[MAXVAL*(a-mean)/std for a in seq_d] 
    return normSeq