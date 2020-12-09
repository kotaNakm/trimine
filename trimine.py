""" Python implementation of TriMine @ KDD'12 """

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, loggamma
from tqdm import trange
import numba
import copy
from hmmlearn.hmm import GaussianHMM

MINK = 1
MAXK = 8

# https://github.com/scikit-learn/scikit-learn/blob/7e85a6d1f/sklearn/decomposition/_online_lda.py#L135

class TriMine(object):
    def __init__(self, k, u, v, n, outputdir):
        # statuses
        self.k = k  # of topics
        self.u = u  # of objects
        self.v = v  # of actors
        self.n = n  # data duration
        self.outputdir = outputdir
        self.train_log = []
        self.max_alpha = 1#100 #0.001
        self.max_beta  = 1#10
        self.max_gamma = 1#10
        self.init_params()


    def init_params(self, alpha=None, beta=None, gamma=None):
        """ Initialize model parameters """

        # if parameter > 1: pure
        # if parameter < 1: mixed
        self.alpha = 0.5/self.k #50/self.k #0.0005/self.k  #self.u
        self.beta  = 0.1 #5  #self.v
        self.gamma = 0.1 #5 #self.n

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
        

    def update_status(self,tensor,pre_n):
        tmp_Nkn = self.Nkn
        tmp_Z = self.Z
        pre_Nsum = self.Nsum
        self.Nsum += tensor.sum()

        self.Nkn = np.zeros((self.k, self.n), dtype=int)
        self.Z = np.full((self.Nsum), -1)

        self.Nkn[:,:pre_n] = tmp_Nkn
        self.Z[:pre_Nsum] = tmp_Z

    def get_params(self, **kwargs):
        return self.alpha, self.beta, self.gamma


    def get_factors(self):
        return self.O, self.A, self.C


    def infer(self, tensor, n_iter=10, tol=1.e-8,
              init=True, verbose=True):
        """
        Given: a tensor (actors * objects * time)
        Find: matrices, O, A, C
        """
        if init == True:
            self.init_status(tensor)

        for iteration in range(n_iter):
            # print(self.Nku)
            # Sampling hidden topics z, i.e., Equation (1)
            self.Z = self.sample_topic(tensor, self.Z, 0,0)
            
            # Update parameters
            self.update_alpha()
            self.update_beta()
            self.update_gamma()
            self.compute_factors(0)

            # Compute log-likelihood
            llh = self.loglikelihood()
            self.train_log.append(llh)

            # Early break
            if iteration > 0:
                if np.abs(self.train_log[-1] - self.train_log[-2]) < tol:
                    print('Early stopping')
                    break
            if verbose == True:
                # Print learning log
                print('Iteration', iteration + 1)
                print('loglikelihood=\t', llh)
                print(f'| alpha\t| {self.alpha:.3f}')
                print(f'| beta \t| {self.beta:.3f} ')
                print(f'| gamma\t| {self.gamma:.3f}')
                # Save learning log
                plt.plot(self.train_log)
                plt.xlabel('Iterations')
                plt.ylabel('Log-likelihood')
                plt.savefig(self.outputdir + 'train_log.png')
                plt.close()
        

    def infer_online(self, tensor, n_iter=10, tol=1.e-8,verbose=True):
        """
        Given: a tensor (actors * objects * time)
        Find: matrices, O, A, C
        """
        if not tensor.ndim == 3 :
            tensor = tensor[:,:,np.newaxis] 

        u,v,n = tensor.shape
        pre_n = self.n
        self.n += n
        cnt = self.Nsum
        self.update_status(tensor,pre_n)

        for iteration in range(n_iter):
            # Sampling hidden topics z, i.e., Equation (1)
            self.Z = self.sample_topic(tensor, self.Z, pre_n,cnt)

            # Update parameters
            self.update_alpha()
            self.update_beta()
            self.update_gamma()

            # Early break
            if iteration > 0:
                if np.abs(self.train_log[-1] - self.train_log[-2]) < tol:
                    print('Early stopping')
                    break
                    self.compute_factors(pre_n)

        self.compute_factors(pre_n)
        # Compute log-likelihood
        llh = self.loglikelihood()
        self.train_log.append(llh)

        if verbose == True:
            # Print learning log
            # print('Iteration', iteration + 1)
            print('loglikelihood=\t', llh)
            print(f'| alpha\t| {self.alpha:.3f}')
            print(f'| beta \t| {self.beta:.3f} ')
            print(f'| gamma\t| {self.gamma:.3f}')
            # Save learning log
            plt.plot(self.train_log)
            plt.xlabel('Iterations')
            plt.ylabel('Log-likelihood')
            plt.savefig(self.outputdir + 'train_log.png')
            plt.close()
    

    def loglikelihood(self):
        """ Compute Log-likelihood """

        # Symmetric dirichlet distribution
        # https://en.wikipedia.org/wiki/Dirichlet_distribution

        llh = 0
        llh = loggamma(self.alpha * self.k) - self.k * loggamma(self.alpha)
        # llh += loggamma(self.alpha * self.u) - self.u * loggamma(self.alpha)
        llh += loggamma(self.beta * self.k) - self.k * loggamma(self.beta)
        llh += loggamma(self.gamma * self.k) - self.k * loggamma(self.gamma)

        for i in range(self.k):
            llh += (self.alpha - 1) * sum([np.log(self.O[j, i]) for j in range(self.u)]) / self.u
            llh += (self.beta  - 1) * sum([np.log(self.A[j, i]) for j in range(self.v)]) / self.v
            llh += (self.gamma - 1) * sum([np.log(self.C[j, i]) for j in range(self.n)]) / self.n

        return llh

    def sample_topic(self, X, Z, pre_n,cnt):
        return _sample_topic(self.Nk,self.Nu,self.Nku,self.Nkv,self.Nkn,self.k,self.u,self.v,self.n,self.alpha,self.beta,self.gamma,X,Z,pre_n,cnt)

    def update_alpha(self):
        # https://www.techscore.com/blog/2015/06/16/dmm/
        num = -1 * self.u * self.k * digamma(self.alpha)

        den = -1 * self.u * self.k * digamma(self.alpha * self.u)
        for i in range(self.k):
            den += self.u * digamma(self.Nk[i] + self.alpha * self.u)
            for j in range(self.u):
                num += digamma(self.Nku[i, j] + self.alpha)

        # den = -1 * self.u * self.k * digamma(self.alpha * self.k)
        # for i in range(self.u):
        #     den += self.k * digamma(self.Nu[i] + self.alpha * self.k)
        #     for j in range(self.k):
        #         num += digamma(self.Nku[j, i] + self.alpha)

        self.alpha *= num / den

        if self.alpha > self.max_alpha:
            self.alpha = self.max_alpha
        if self.alpha < 1.e-8:
            self.alpha = 1.e-8


    def update_beta(self):
        num = -1 * self.k * self.v * digamma(self.beta)
        den = -1 * self.k * self.v * digamma(self.beta * self.v)

        for i in range(self.k):
            den += self.v * digamma(self.Nk[i] + self.beta * self.v)
            for j in range(self.v):
                num += digamma(self.Nkv[i, j] + self.beta)

        self.beta *= num / den

        if self.beta > self.max_beta:
            self.beta = self.max_beta


    def update_gamma(self):
        num = -1 * self.k * self.n * digamma(self.gamma)
        den = -1 * self.k * self.n * digamma(self.gamma * self.n)

        for i in range(self.k):
            den += self.n * digamma(self.Nk[i] + self.gamma * self.n)
            for j in range(self.n):
                num += digamma(self.Nkn[i, j] + self.gamma)

        self.gamma *= num / den

        if self.gamma > self.max_gamma:
            self.gamma = self.max_gamma


    def compute_factors(self,pre_n):
        """ Generate three factors/matrices, O, A, and C
        """
        
        # self.O = np.zeros((self.u, self.k))
        # self.A = np.zeros((self.v, self.k))
        # self.C = np.zeros((self.n, self.k))

        if pre_n:
            tmp_C = copy.deepcopy(self.C)
            self.C = np.zeros((self.n, self.k))
            self.C[:pre_n,:] = tmp_C

        for i in range(self.k):
            for j in range(self.u):
                self.O[j, i] = (
                    (self.Nku[i, j] + self.alpha)
                    / (self.Nu[j] + self.alpha * self.k))
            for j in range(self.v):
                self.A[j, i] = (
                    (self.Nkv[i, j] + self.beta)
                    / (self.Nk[i] + self.v * self.beta))
            for j in range(pre_n,self.n):
                self.C[j, i] = (
                    (self.Nkn[i, j] + self.gamma)
                    / (self.Nk[i] + self.n * self.gamma))

        # print(self.O.sum(axis=1))
        # print(self.A.sum(axis=0))
        # print(self.C.sum(axis=0))
        return self.O, self.A, self.C


    def save_model(self):
        """ Save all of parameters for TriMine
        """
        with open(self.outputdir + 'params.txt', 'a') as f:
            f.write(f'topic,{self.k}\n')
            f.write(f'alpha,{self.alpha}\n')
            f.write(f'beta, {self.beta}\n')
            f.write(f'gamma,{self.gamma}\n')

        np.savetxt(self.outputdir + 'O.txt', self.O)
        np.savetxt(self.outputdir + 'A.txt', self.A)
        np.savetxt(self.outputdir + 'C.txt', self.C)
        np.savetxt(self.outputdir + 'train_log.txt', self.train_log)


@numba.jit #(nopython=True)
def _sample_topic(Nk,Nu,Nku,Nkv,Nkn,k,u,v,n,alpha,beta,gamma,X,Z,pre_n,cnt):
    """
    X: event tensor
    Z: topic assignments of the previous iteration
    """

    Nu = X.sum(axis=(1, 2))
    # for t in trange(self.n, desc='#### Infering Z'):

    cnt=cnt
    for t in range(pre_n,n):
        for i in range(u):
            for j in range(v):
                # for each non-zero event entry,
                # assign latent topic, z
                count =  X[i, j, t-pre_n]
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
                        #     exit()
                    """ compute posterior distribution """
                    posts = np.zeros(k)
                    # print(self.Nku[:, i])
                    # print(self.Nkv[:, j])
                    # print(self.Nkn[:, t])
                    for r in range(k):
                        # NOTE: Nk[r] = Nkv[r, :].sum() = Nkn[r, :].sum()
                        O = A = C = 1
                        O = (Nku[r, i] + alpha) / (Nu[i] + alpha * k)
                        A = (Nkv[r, j] + beta) / (Nk[r] + beta  * v)
                        C = (Nkn[r, t] + gamma) / (Nk[r] + gamma * n)
                        # print(O.shape,A.shape,C.shape)
                        # posts[r] = float('{:.5g}'.format(O * A * C))
                        posts[r] = O * A * C
                    posts = posts /(posts.sum()*1.1) # normalize with bias
                    if posts.sum()>1 or posts.sum()<0:
                        print(posts)
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

# @numba.jit #(nopython=True)
# def _online_sample_topic(Nk,Nu,Nku,Nkv,Nkn,k,u,v,n,alpha,beta,gamma,X,Z,pre_n):
#     """
#     X: event tensor
#     Z: topic assignments of the previous iteration
#     """
#     Nu = X.sum(axis=(1, 2))
#     print(Nu.shape)
#     exit()
#     # for t in trange(self.n, desc='#### Infering Z'):
#     for t in range(pre_n,n):
#         for i in range(u):
#             for j in range(v):
#                 # for each non-zero event entry,
#                 count = X[i, j, t]
#                 for e in range(count):    
#                     # assign latent topic, z
#                     topic = Z[i, j, t]
#                     if count == 0:
#                         continue
#                     if not topic == -1:
#                         Nk[topic] -= count
#                         Nku[topic, i] -= count
#                         Nkv[topic, j] -= count
#                         Nkn[topic, t] -= count
#                         # if ((Nk  < 0).sum() > 0 or
#                         #     (Nkv < 0).sum() > 0 or
#                         #     (Nku < 0).sum() > 0 or
#                         #     (Nkn < 0).sum() > 0):
#                         #     print("Invalid counter N has been found")
#                         #     # print(self.Nk,self.Nkv,self.Nku,self.Nkn)
#                         #     exit()
#                     """ compute posterior distribution """
#                     posts = np.zeros(k)
#                     # print(self.Nku[:, i])
#                     # print(self.Nkv[:, j])
#                     # print(self.Nkn[:, t])
#                     for r in range(k):
#                         # NOTE: Nk[r] = Nkv[r, :].sum() = Nkn[r, :].sum()
#                         O = A = C = 1
#                         O = (Nku[r, i] + alpha) / (Nu[i] + alpha * k)
#                         A = (Nkv[r, j] + beta) / (Nk[r] + beta  * v)
#                         C = (Nkn[r, t] + gamma) / (Nk[r] + gamma * n)
#                         # print(O.shape,A.shape,C.shape)
#                         posts[r] = O * A * C
#                     posts = posts / posts.sum()  # normalize
                    
#                     topic = np.argmax(np.random.multinomial(1, posts))
#                     # print(topic, '<-', posts)
#                     Z[i, j, t] = topic
#                     Nk[topic] += count
#                     Nku[topic, i] += count
#                     Nkv[topic, j] += count
#                     Nkn[topic, t] += count
#     return Z