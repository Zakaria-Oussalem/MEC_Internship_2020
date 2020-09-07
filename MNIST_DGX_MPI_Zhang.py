import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import time
torch.set_default_dtype(torch.float32)
#import cProfile


from mpi4py import MPI





class network(nn.Module):
    """Fully Connected Neural Network"""
    def __init__(self,
            bias=None,
            pop_size=None,
            k_val=None,
            cross_prob=None,
            rank=0,
            gpu=False
            ):
        """Initialize Parameters

        Args:
            bias (Defualt=None): Use bais value or not
        """
        super().__init__()

        # Parameters
        self.bias = bias
        self.num_layers = 0
        self.rank = rank
        self.gpu = gpu

        # DE parameters
        self.pop_size = pop_size
        self.k_val = k_val
        self.cross_prob = cross_prob
        self.best = 0
        self.L = []
        self.F = 0
        self.bounds = [-30.0,30.0]

        

        # DE candidate values
        self.population = None

        # Initialize the layers of the neural network
        #
        # torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
        # stride=1, padding=0, output_padding=0, groups=1, bias=True,
        # dilation=1, padding_mode='zeros')
        #
        # NOTE: Because PyTorch does not support 'same' padding for ConvTranspose3d
        #       you will need to compute the padding and the output_padding by yourself.
        #       Use the below function for the same:
        #       o = (i -1)*s - 2*p + k + output_padding
        #
        # Use ConvTranspose3dSame when stride == 1, padding = 'same'

#        self.layers = nn.Sequential(
#            nn.Conv2d(1, 3, kernel_size=9),
#            nn.ReLU(),
#            nn.Conv2d(3, 3, kernel_size=10),
#            nn.ReLU(),
#            nn.Conv2d(3, 10, kernel_size= 11),
#            nn.Softmax(dim = 1)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=7),
            nn.ReLU(),
            nn.Conv2d(3, 5, kernel_size=15),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size= 8),
            nn.Softmax(dim = 1))
        
        self.num_layers = 6
        self.shape = 0
        for i in range(self.num_layers):
            if i%2 == 0 :
                self.shape += np.prod(self.layers[i].weight.shape) + self.layers[i].bias.shape
    
        self.shape = self.shape[0]
        print("The number of parametres : ",self.shape)
        self.past_loss = np.zeros(self.pop_size)

    def forward(self, x):
        with torch.no_grad():
            x = self.layers(x)
        x_ = x.flatten(1) 
        return x_
    def _objective(self,output, expected,device):
        """Compute MSE between output and expected value
        Args:
            output: Output from model
            expected: Expected output
        Returns:
            MSE error as tensor
        """
        #
        #output = output.to(device = expected.device) #
        
        output = torch.where(output == 0,self.saf,output)
        #output = torch.where(output == 1,torch.tensor([0.999999],device = device),output)
        out1 = torch.mul(-expected,torch.log(output))
        #out2 = torch.mul(expected-1,torch.log(1 - output))
        #out = torch.add(out1,out2)
        return torch.sum(out1, dim = 1)


    def set_weights_to_layers(self, candidate):
        """Set model weights and bias value as candidates value

        Args:
            candidate: Candidate tensor
        """
        last_idx = 0

        # Iterate over every layer
        for layer_idx in range(0, self.num_layers, 1):
            if layer_idx %2 == 0 : 
                # Get layer dimensions
                w_shape = self.layers[layer_idx].weight.shape
                w_numel = np.prod(w_shape)
                if self.bias :
                    b_shape = self.layers[layer_idx].bias.shape
                    b_numel = np.prod(b_shape)
                # Decode the candidate and get weight, bias matrices
                weight = candidate[last_idx:last_idx + w_numel].reshape(w_shape)
                last_idx += w_numel
                if self.bias :
                    bias = candidate[last_idx:last_idx + b_numel].reshape(b_shape)
                    last_idx += b_numel
                else :
                    bias = []

#                if self.gpu:
#                    weight = weight.to(device='cuda:'+str(model.rank))
#                    if self.bias:
#                        bias = bias.to(device='cuda:'+str(model.rank))
#                # Set layer weight, bias
                self.layers[layer_idx].weight = torch.nn.Parameter(weight)
                if self.bias :
                    self.layers[layer_idx].bias = torch.nn.Parameter(bias)
   

    def _mutant(self, idx, F):
        """Generate Mutant vector and perform Crossover
        Args:
            idx: Index of candidate
            F: F value hyperparameter
        Returns:
            mutant: Generated mutant vector
        """
        # Generate random indices
        r = torch.randint(self.pop_size, (3,))
        # Re-generate if it contains candidate index
        while r[1] == r[0] or r[2] == r[0] or r[2] == r[1] or (idx in r):
            r = torch.randint(0, self.pop_size, (3,))
        

        # Compute mutant
        mutant = self.population[r[0]] + \
            self.k_val * (self.population[self.best] - self.population[r[0]]) + \
            F * (self.population[r[2]] - self.population[r[1]])
        #mutant = mutant.to(self.device)
        # Crossover
        probs = torch.rand(mutant.shape[0],device = self.device)
        
        return torch.where(probs >= self.cross_prob,self.population[idx],mutant)
        
        
        
#        r = np.random.randint(0, self.pop_size, (3,))
#        # Re-generate if it contains candidate index
#        while r[1] == r[0] or r[2] == r[0] or r[2] == r[1]:
#            r = np.random.randint(0, self.pop_size, (3,))
#
#        # Compute mutant
#        mutant = np.zeros(self.population[idx].shape)
#        mutant = self.population[idx] + \
#            self.k_val * (self.population[self.best] - self.population[idx]) + \
#            F * (self.population[r[2]] - self.population[r[1]])
#
#        # Crossover
#        probs = np.random.rand(mutant.shape[0])
#        return np.where(
#            probs >= self.cross_prob,
#            self.population[idx],
#            mutant)

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))

    def local_search(self,iteration,maxiter,input_, expected):
        # 1 & 2
        '''
        JN input parameter:
        iteration  = current generation
        maxiter    = total number of genrations
        input_     = input data
        expected   = output data
        Returns : list of losses of each canidate in the population
        '''
        
        pop_scores =[]
        for i in range(self.pop_size) :
            pop_scores.append([self.population[i],self.past_loss[i],i])
        #JN 19/5 ordering the candidates in the descending order of their losses
        pop_scores.sort(key=lambda x: x[1],reverse=True)
        
        #this variable is used instaead of the self.population later the self.population will be repaced by asc_pop
        #asc_pop = [obj[0] for obj in pop_scores]
        
        # 3: crossover
        for i in self.L[:-1]:
            X_i, X_ip1 = pop_scores[i][0][:], pop_scores[i+1][0][:]
            r = torch.rand((1,1)).item()
            X_i_new=X_i+ r* (X_ip1 - X_i)

            #JN 19/5 new candidate loss
            self.set_weights_to_layers(X_i_new)
            #we predict the output corresponding to the candidate
            vec_output = self.forward(input_)
            #Now we calculate the loss corresponding to the candidate
            #As you see we are using the linear interpolation the professor talked about
            
            loss_new = torch.mean(self._objective(vec_output, expected,self.device)).item()
           

            
            loss_next = self.past_loss[pop_scores[i+1][2]]

            #JN 19/5 selecting the best candidate from old and new canidate
            if loss_new<loss_next:
                self.population[pop_scores[i][2]]=X_i_new[:]
                self.past_loss[pop_scores[i][2]] = loss_new
                pop_scores[i][1] == loss_new

        
        

        pop_scores.sort(key=lambda x: x[1],reverse=True)
        p_m = [(self.pop_size - i + 1.0)/self.pop_size for i in range(self.pop_size)]
        
        #asc_pop=sorted(asc_pop,key=lambda cand:cost_func(decode_1(cand,rand_keys,agent_wid_rand_key_dict).tolist(),nss,costs_of_agents_dict)[0],reverse=True)
        #gen_peanlty=[]
        # 5 Nonuniform mutation operation
    
        cont_term=np.exp((-2.0*(iteration%50))/(1.0*50))*self.sigmoid(((50/2.0)-iteration%50)/1.0)
        for i in self.L:
           # p_m_i=(self.pop_size-i)/(self.pop_size)

#            if i == 0 : 
#                start = time.time()
            r = torch.rand((self.shape,),device = self.device)
            X_i_new= self.population[pop_scores[i][2]][:]
            X_i_new_h = X_i_new + (self.bounds[1] - X_i_new)*r*p_m[i]*cont_term
            X_i_new_b = X_i_new + (X_i_new - self.bounds[0])*r*p_m[i]*cont_term
            probs = torch.randint(2,(self.shape,),device = self.device)
            X_i_new = torch.where(probs == 0 , X_i_new_h, X_i_new_b)
            X_i_new = torch.where((self.population[pop_scores[i][2]]< self.bounds[1])*(self.population[pop_scores[i][2]]>self.bounds[0]),X_i_new,self.population[pop_scores[i][2]])

#            for j in range(self.shape):
#                #JN do this operation only when the dimension value is between bounds
#                if self.population[i][j]<self.bounds[1] and self.population[i][j]>self.bounds[0] :
#                    b=torch.randint(1,(1,1)).item()
#                    r= torch.rand((1,1)).item()
#                    if b==0:
#                        X_i_new[j]=self.population[i][j]+(self.bounds[1]-self.population[i][j])*r*cont_term
#                    else:
#                        X_i_new[j]=self.population[i][j]+(self.population[i][j]-self.bounds[0])*r*cont_term
#                else :
#                    X_i_new[j] = self.population[i][j] 
#            if i == 0:
#                end = time.time()
#                print("time on the sorting : ", end - start)

            #JN loss calculaiton for the new candidates
            self.set_weights_to_layers(X_i_new)
            #we predict the output corresponding to the candidate
            vec_output = self.forward(input_)
            #Now we calculate the loss corresponding to the candidate
            #As you see we are using the linear interpolation the professor talked about
            loss_new = torch.mean(self._objective(vec_output, expected,self.device)).item()


           
            loss_pres = self.past_loss[pop_scores[i][2]]
            


            if loss_new<loss_pres:
                self.population[pop_scores[i][2]] = X_i_new[:]
                self.past_loss[pop_scores[i][2]] = loss_new
        return [pop_scores[j][2] for j in self.L]
        
                
    def backwards_de(self, input_, expected, idx):
        """Backwards pass on Neural Network using Differential Evolution
        Args:
            vec_output: Forward pass output
            input_ : Input Matrix
            expected: Expected value matrix
        """


        trial = self._mutant(idx, self.F)
        self.set_weights_to_layers(trial)
        vec_output = self.forward(input_)
        trial_loss = torch.mean(self._objective(vec_output, expected,self.device)).item()
        
        if trial_loss <= self.past_loss[idx] :
            self.population[idx] = trial[:]
            self.past_loss[idx] = trial_loss

    def best_apply(self,X_test,y_test):
            loss_best = 10**5
            index_best = 0
            for i in range(self.pop_size):
                self.set_weights_to_layers(self.population[i])
                vec_output = self.forward(X_test)
                loss = torch.mean(self._objective(vec_output, y_test,self.device)).item()
                if loss <= loss_best :
                    loss_best = loss
                    index_best = i
            self.set_weights_to_layers(self.population[index_best])

def candidate_loss(model_g, X_test, Y_test):
    torch.cuda.synchronize(device=None)
    loss = torch.zeros(((len(model_g.L)),2), device='cuda:'+str(model_g.rank))
    for idx, val in enumerate(model_g.L):
        model_g.set_weights_to_layers(model_g.population[val])
        output = model_g(X_test)
        
        loss[idx][0] = torch.mean(0.5 * (Y_test - output).pow(2))
        loss[idx][1] = val

    loss = loss.cpu().numpy()
    return loss
        


def dataset():
    
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")
    
    index = np.arange(60000)
    np.random.shuffle(index)
    X_train =[X_train[i] for i in index[:60000]]
    y_train =[y_train[i] for i in index[:60000]]
    index = np.arange(10000)
    np.random.shuffle(index)
    X_test =[X_test[i] for i in index[:10000]]
    y_test =[y_test[i] for i in index[:10000]]   
    
    
    X_test,y_test,X_train,y_train  = torch.tensor(X_test),torch.tensor(y_test),torch.tensor(X_train),torch.tensor(y_train)
    X_test = X_test.unsqueeze(1)
    X_train = X_train.unsqueeze(1)

    return X_train,y_train,X_test,y_test

def train_network(X_train,
                  y_train,
                  X_test,
                  y_test,
                  comm,
                  rank,
                  model,
                  epochs=100
                  ):
    """Training loop
    Args:
        X_train : features for training
        X_test : features for testing
        y_train : labels for training
        y_test : labels for testing
        model: Neural Network object (CPU)
        model_g: Neural Network object (GPU)
        epoch (Default=100): Number of epochs
        batch_size (Default=1): Batch Size
    Returns:
        loss_history: Array of training loss
    """
    if rank == 0 :
        
        loss_history_best = []
        loss_history_average = []
        accuracies = []
        start = time.time()
    else :
        loss_history_best = None
        loss_history_average = None
        accuracies = None
        

    for i in range(epochs):
#        
#        if i%10 != 0 :
#            index = torch.randperm(60000)
#            X_train_ = X_train[index[:12000]]
#            y_train_ = y_train[index[:12000]]
#        else :


        if rank == 0 :
            F = np.random.uniform(-2,2)
            if i % 50 == 0 :
                acc = scores(model,X_train,y_train, X_test,y_test,True)
                accuracies.append(acc)
        else :
            F = None
                           
        
        model.F = comm.bcast(F, root = 0)

        
        for _, val in enumerate(model.L):
            if i == 0 :
                model.set_weights_to_layers(model.population[val])
                vec_output = model.forward(X_train)
                model.past_loss[val] = torch.mean(model._objective(vec_output, y_train,model.device)).item()
            model.backwards_de(X_train, y_train, val)
        
        comm.barrier()
        
        Population = comm.gather([model.population[i].cpu().numpy() for i in model.L], root = 0)
        model.past_loss = comm.gather([model.past_loss[i] for i in model.L], root = 0)
        
        comm.barrier()
        
        if rank == 0 :
            Population = [member for sublist in Population for member in sublist]
            model.past_loss = [member for sublist in model.past_loss for member in sublist]
         
        comm.barrier()
        
        model.population = comm.bcast(Population, root = 0)
        model.population = torch.tensor(model.population).to(device = model.device)
        model.past_loss = comm.bcast(model.past_loss, root = 0)
        
        comm.barrier()
            
        lol = model.local_search(i,epochs,X_train,y_train)
        
        comm.barrier()
                
        Population = comm.gather([model.population[i].cpu().numpy() for i in lol], root = 0)
        model.past_loss = comm.gather([model.past_loss[i] for i in lol], root = 0)
        
        comm.barrier()    

        
        if rank == 0 :
            Population = [member for sublist in Population for member in sublist]
            model.past_loss = [member for sublist in model.past_loss for member in sublist]
            indexo = np.argmin(model.past_loss) 
            best_loss = model.past_loss[indexo]
            avr = sum(model.past_loss) / float(len(model.past_loss))
            loss_history_average.append(avr)
            loss_history_best.append(best_loss)
            print('Epoch: {} Best loss: {} Mean loss: {}'.format(i, best_loss,avr))
        else :
            indexo = None
                    
        comm.barrier()


        
        model.population = comm.bcast(Population, root = 0)
        model.population = torch.tensor(model.population).to(device = model.device)
        model.past_loss = comm.bcast(model.past_loss, root = 0)
        model.best = comm.bcast(indexo, root = 0)
            
        comm.barrier()
    
    if rank == 0:
        end = time.time()
        print("time is   :  ",end-start)

    return loss_history_best, loss_history_average, np.array(accuracies)

def scores(model,X_train,y_train, X_test,y_test,inside):
    """Compute MAPE and RSQ.
    Args:
        model: Neural Network object
        X_test: features for testing
        y_test: labels fro testing
        sc_y: labels scaler object
        path: location to save predicted labels
    """
    
    model.best_apply(X_test,y_test)
    y_pred = model.forward(X_test).cpu()
    y_pred = np.asarray(torch.argmax(y_pred, dim = 1).squeeze())
    y_test_ = np.asarray(torch.argmax(y_test.cpu(), dim = 1))
    acc1 = accuracy_score(y_test_,y_pred)
    model.best_apply(X_train,y_train)
    y_pred = model.forward(X_train).cpu()
    y_pred = np.asarray(torch.argmax(y_pred, dim = 1).squeeze())
    y_train_ = np.asarray(torch.argmax(y_train.cpu(), dim = 1))
    acc2 = accuracy_score(y_train_,y_pred)
    if inside :
        print("###### Results ##########")
        print("The Validation Acuuracy is : ",acc1)
        print("The Training Acuuracy is : ",acc2)
    return [acc1,acc2]



def Main():
    comm = MPI.COMM_WORLD
    
    # Differential Evolution Parameters
    POP_SIZE = 200
    K_VAL = 0.5
    CROSS_PROB = 0.9

    # Neural network parameters
    epochs = 5000

    # Initializing the parallelisation params
    rank = comm.Get_rank()
    size = comm.Get_size()
    perrank = POP_SIZE//size

    #if rank == 0:
    #    assert size == torch.cuda.device_count()

    # Set device for MPI thread
    torch.cuda.device(rank)

    # Model for each candidate
    model = network(True, POP_SIZE, K_VAL, CROSS_PROB, rank, True)
    # INPUT SHAPE: a = torch.rand((1, 1, 8, 3, 11))

    # Model copy on the GPU
    #model_g = network(True, POP_SIZE, K_VAL, CROSS_PROB, rank, True)

    model.rank = rank
    #model_g.rank = rank

    if rank == 0:
        X_train,y_train,X_test,y_test = dataset()
    else :
        X_train,y_train,X_test,y_test = None,None,None,None

    X_train = comm.bcast(X_train, root = 0)
    y_train = comm.bcast(y_train, root = 0)
    X_test = comm.bcast(X_test, root = 0)
    y_test = comm.bcast(y_test, root = 0)
    
    X_test = X_test.to(device='cuda:'+str(model.rank))
    y_test = y_test.to(device='cuda:'+str(model.rank))
    X_train = X_train.to(device='cuda:'+str(model.rank))
    y_train = y_train.to(device='cuda:'+str(model.rank))

    
    if rank == 0 :
        population = torch.cat((torch.rand((
                model.pop_size//4, model.shape))-0.5,
                                          torch.normal(
                0,1,size = (model.pop_size//4,model.shape)),
                                          torch.rand((
                model.pop_size//4, model.shape))*0.5-0.25,
                                          torch.normal(
                0,0.5,size = (model.pop_size//4,model.shape))),
                       dim = 0)
#        population = torch.from_numpy(np.load("weights_mnist_r_exp_cnn_6740_9000.npy"))
        

    else :
        population = None
    
    comm.barrier()

    model.population = comm.bcast(population, root = 0)
    #model_g.population = model.population
    model.population = model.population.to(device='cuda:'+str(model.rank))
    model.device = 'cuda:'+str(model.rank)
    model.saf = torch.tensor([1e-45],device = model.device)


    # Parallelizing the candidates
    sep = []
    if rank == 0 :
        cte = 0
        rest = model.pop_size - perrank*size
        for i in range(size):
            sep.append([j for j in range(cte+i*perrank,cte+(i+1)*perrank)])
            if rest > 0 :
                sep[i].append((i+1)*perrank+cte)
                cte += 1
                rest = rest - 1
    else :
        sep = None
    L = comm.bcast(sep, root = 0)[rank]
            



    print('rank: {}, perrank: {}, size: {}, L: {}\n'.format(rank, perrank, size, L))

    model.L = L
    #model_g.L = L

    comm.barrier()
  
    
    # Train
    loss_history_best, loss_history_average,accuracies = train_network(X_train, y_train, X_test, y_test, comm, rank, model,
                                 epochs)
    
    
#    model.best_apply(X_train,y_train)
#    index = model.best
#    maxi = np.max(model.population[index].cpu().numpy())
#    mini = np.min(model.population[index].cpu().numpy())
#    print("maxi is ",maxi)
#    print("mini is ",mini)
    comm.barrier()

    if rank == 0 :
        np.save("weights_mnist_pmi_cnn_6740_5000.npy",model.population.cpu().numpy())
        np.save("accuracies_mnist_pmi_cnn_6740_5000.npy",accuracies)
#        np.save("loss_best_MNIST_gpu_mpi_4700 3000.npy",loss_history_best)
#        np.save("loss_average_MNIST_gpu_mpi_4700 3000.npy",loss_history_average)


        # Plot loss history
#        plt.plot(range(0,3000),loss_history_best, color = 'red', label = 'best')
#        plt.plot(range(0,3000),loss_history_average, color = 'blue' , label = 'average')
#        plt.title('The Validation train loss in each gen ')
#        plt.xlabel('Generation')
#        plt.ylabel('Validation Loss')
#        plt.legend()
#        plt.savefig("Mnist_gpu_mpi_4700 convergence 3000  ")
#        plt.close()
#        
#        plt.plot(range(100,6000),loss_history_best[100:], color = 'red', label = 'best')
#        plt.plot(range(100,6000),loss_history_average[100:], color = 'blue' , label = 'average')
#        plt.title('The Validation train loss in each gen')
#        plt.xlabel('Generation')
#        plt.ylabel('Validation Loss')
#        plt.legend()
#        plt.savefig("Mnist_gpu_mpi_new_new convergence 6000 details")
#        plt.close()
        
#        plt.plot(range(0,3000,50),accuracies[:,0], color = 'red', label = 'Test')
#        plt.plot(range(0,3000,50),accuracies[:,1], color = 'blue' , label = 'Train')
#        plt.title('Evolution of accuracies')
#        plt.xlabel('Generation')
#        plt.ylabel('Accuracy')
#        plt.legend()
#        plt.savefig("Mnist_gpu_mpi_4700 Accuracy convergence 3000")
#        acc1,acc2 = scores(model,X_train,y_train, X_test,y_test,False)
#        print("###### Results ##########")
#        print("The Final Validation Acuuracy is : ",acc1)
#        print("The Final Training Acuuracy is : ",acc2)
        


Main()