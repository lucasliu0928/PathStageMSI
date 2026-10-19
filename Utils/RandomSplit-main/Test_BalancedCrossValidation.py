# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# August 2023
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import numpy as np
import sys
sys.path.insert(0, '../Utils/RandomSplit-main/')
from RandomSplit import BalancedCrossValidation


from RandomSplit import MakeBalancedCrossValidation
import random
import numpy as np

#Example1 
label_df = tile_info_df_pt[["SAMPLE_ID","AR","HR2","PTEN","RB1","TP53","TMB_HIGHorINTERMEDITATE","MSI_POS"]]
label_df.index = label_df['SAMPLE_ID']
label_df.drop(columns = 'SAMPLE_ID', inplace = True)
check_w  = label_df.T
ids = label_df.index
check_w = np.array(check_w)
# Remove columns that are all zeros
check_w = check_w[:, ~np.all(check_w == 0, axis=0)]
column_map = {i: ids[i] for i in range(check_w.shape[1])}


#Example
K = check_w.shape[0]
N = check_w.shape[1]
F=5
p_test=0.25

# Randomly assign each sample to a patient


# np.random.seed(0)
# W = np.random.randint(size=[K,N], low=0, high=2)

training_lists, validation_lists, test_list, res, res_test = MakeBalancedCrossValidation(check_w, F, column_map, testing_size=p_test, tries=10)

#Example2

K = 7
N = 20
F=5
p_test=0.25

random.seed(42)
patient_ids = [f"ID{i+1}" for i in range(N)]
column_map = {i: patient_ids[i] for i in range(N)}

np.random.seed(0)
W = np.random.randint(size=[K,N], low=0, high=2)


training_lists, validation_lists, test_list, res, res_test = MakeBalancedCrossValidation(W, F, column_map, testing_size=p_test, tries=10)




def PureRandomCrossValidation(W, F, tries=10, aggregator=np.max):
    assert W.ndim == 2
    
    N = W.shape[1]

    assert F > 1 and F <= N

    assert np.all(W.max(axis=0) > 0) # Make sure all instances count for something
   
    # Remove rows with no counts over any instance
    D = W.sum(axis=1)
    W = W[D > 0, :]
    D = D[D > 0]
    
    K = W.shape[0]
    
    assert K > 1 and N >= K
    
    D = 1.0/D
    Z = np.eye(K) - 1.0/K
    
    # This is the same as D*W... just in numpy weirdness
    W = W*D[..., None]
    
    # This is ZDW
    W = Z @ W
    
    bestRes = None
    bestFolds = None
    
    ind = np.arange(N)
    
    for _ in range(tries):        
        np.random.shuffle(ind)

        res = []
        folds = []

        for f in range(F):
            val_begin = N*f//F
            val_end = N*(f+1)//F
        
            x = np.ones(N, dtype=int)
            x[ind[val_begin:val_end]] = 0
       
            folds.append(x)
            res.append(np.linalg.norm(np.inner(W, x)))
        
        if bestRes is None or aggregator(res) < aggregator(bestRes):
            bestRes = res
            bestFolds = folds
            
    return bestFolds, bestRes

def RunBenchmark():
    K = 11
    N = 200
    F = 3
    numRuns=1 #100000
    tries=1
    aggregator=np.max
    
    np.random.seed(727)
    seeds = np.random.randint(size=numRuns, low=1, high=2**31-1)

    allRes = np.zeros(numRuns)
    allResRandom = np.zeros(numRuns)

    ind = np.arange(N)
    
    M = int(0.9*N)
    
    for i in range(numRuns):
        np.random.seed(seeds[i])
        W = np.random.randint(size=[K,N], low=1, high=10)
        
        W[0, :] = 1
        W[7, :] = 0 # Test zero row removal support
        
        for k in range(1,K):
            np.random.shuffle(ind)
            W[k, :][ind[:M]] = 0
        
        assert np.all(W.max(axis=0) > 0)
        
        #expected = np.round(((F-1.0)/F)*W.sum(axis=1)).astype(int)
        
        folds, res = BalancedCrossValidation(W, F, tries=tries, aggregator=aggregator)

        #for f, fold in enumerate(folds):
        #    svd = np.inner(W, fold)
        #    print(f"SVD {f}: {svd}")

        #print(f"Expected: {expected}\n")

        allRes[i] = aggregator(res)

        folds, res = PureRandomCrossValidation(W, F, tries=tries, aggregator=aggregator)

        #for f, fold in enumerate(folds):
        #    random = np.inner(W, fold)
        #    print(f"Random {f}: {random}")

        #print(f"Expected: {expected}\n")

        allResRandom[i] = aggregator(res)

    print(f"SVD: {allRes.mean()} +/- {allRes.std()}")
    print(f"Random: {allResRandom.mean()} +/- {allResRandom.std()}")

if __name__ == "__main__":
    RunBenchmark()

