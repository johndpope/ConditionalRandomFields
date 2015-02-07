import numpy as np

class LinearChainCRF(object):
    """
    Linear chain conditional random field. The context window size
    has a radius of 1.
 
    Option ``lr`` is the learning rate.
 
    Option ``dc`` is the decrease constant for the learning rate.
 
    Option ``L2`` is the L2 regularization weight (weight decay).
 
    Option ``L1`` is the L1 regularization weight (weight decay).
 
    Option ``n_epochs`` number of training epochs.
 
    **Required metadata:**
 
    * ``'input_size'``: Size of the input.
    * ``'targets'``:    Set of possible targets.
 
    """
    
    def __init__(self,
                 lr=0.001,
                 dc=1e-10,
                 L2=0.001,
                 L1=0,
                 n_epochs=10):
        self.lr=lr
        self.dc=dc
        self.L2=L2
        self.L1=L1
        self.n_epochs=n_epochs

        # internal variable keeping track of the number of training iterations since initialization
        self.epoch = 0 

    def initialize(self,input_size,n_classes):
        """
        This method allocates memory for the fprop/bprop computations
        and initializes the parameters of the CRF to 0 (DONE)
        """

        self.n_classes = n_classes
        self.input_size = input_size

        # Can't allocate space for the alpha/beta tables of
        # belief propagation (forward-backward), since their size
        # depends on the input sequence size, which will change from
        # one example to another.

        self.alpha = np.zeros((0,0))
        self.beta = np.zeros((0,0))
        
        ###########################################
        # Allocate space for the linear chain CRF #
        ###########################################
        # - self.weights[0] are the connections with the image at the current position
        # - self.weights[-1] are the connections with the image on the left of the current position
        # - self.weights[1] are the connections with the image on the right of the current position
        self.weights = [np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes))]
        # - self.bias is the bias vector of the output at the current position
        self.bias = np.zeros((self.n_classes))

        # - self.lateral_weights are the linear chain connections between target at adjacent positions
        self.lateral_weights = np.zeros((self.n_classes,self.n_classes))
        
        self.grad_weights = [np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes))]
        self.grad_bias = np.zeros((self.n_classes))
        self.grad_lateral_weights = np.zeros((self.n_classes,self.n_classes))
                    
        #########################
        # Initialize parameters #
        #########################

        # Since the CRF log factors are linear in the parameters,
        # the optimization is convex and there's no need to use a random
        # initialization.

        self.n_updates = 0 # To keep track of the number of updates, to decrease the learning rate

    def forget(self):
        """
        Resets the neural network to its original state (DONE)
        """
        self.initialize(self.input_size,self.n_classes)
        self.epoch = 0
        
    def train(self,trainset):
        """
        Trains the neural network until it reaches a total number of
        training epochs of ``self.n_epochs`` since it was
        initialize. (DONE)

        Field ``self.epoch`` keeps track of the number of training
        epochs since initialization, so training continues until 
        ``self.epoch == self.n_epochs``.
        
        If ``self.epoch == 0``, first initialize the model.
        """

        if self.epoch == 0:
            input_size = trainset.metadata['input_size']
            n_classes = len(trainset.metadata['targets'])
            self.initialize(input_size,n_classes)
            
        for it in range(self.epoch,self.n_epochs):
            for input,target in trainset:
                self.fprop(input,target)
                self.bprop(input,target)
                self.update()
        self.epoch = self.n_epochs
        
    def fprop(self,input,target):
        """
        Forward propagation: 
        - computes the value of the unary log factors for the target given the input (the field
          self.target_unary_log_factors should be assigned accordingly)
        - computes the alpha and beta tables using the belief propagation (forward-backward) 
          algorithm for linear chain CRF (the field ``self.alpha`` and ``self.beta`` 
          should be allocated and filled accordingly)
        - returns the training loss, i.e. the 
          regularized negative log-likelihood for this (``input``,``target``) pair
        Argument ``input`` is a Numpy 2D array where the number of
        rows is the sequence size and the number of columns is the
        input size. 
        Argument ``target`` is a Numpy 1D array of integers between 
        0 and no. of classes - 1. Its size is the same as the number of
        rows of argument ``input``.
        """

        # (your code should call belief_propagation and training_loss)
        targetLen = len(target)
        self.target_unary_log_factors = np.zeros((targetLen, self.n_classes))
        
        for y_k in range(0, targetLen):
            activation_left = np.zeros((1, self.n_classes))
            activation_right = np.zeros((1, self.n_classes))
            
            if(y_k > 0):
                activation_left = np.dot(input[y_k - 1], self.weights[0]) #yields 1 X n_classes matrix
            
            #Including Bias at Center
            activation_center = self.bias + np.dot(input[y_k], self.weights[1]) #yields 1 X n_classes matrix
            
            if(y_k < (targetLen-1)):
                activation_right = np.dot(input[y_k + 1], self.weights[2]) #yields 1 X n_classes matrix
                
            self.target_unary_log_factors[y_k] =  activation_left + activation_center + activation_right
        
        # In alpha & beta table, 0th column is for y1, 1st for y2, 2nd for y3, ... (K-1)th for yK
        # All calculations are done in log 
        self.alpha = np.zeros((self.n_classes, targetLen))
        self.beta = np.zeros((self.n_classes, targetLen))
        
        #Now, fill the alpha & beta tables
        self.belief_propagation(input)
        return self.training_loss(target, self.target_unary_log_factors, self.alpha, self.beta)

    def belief_propagation(self,input):
        """
        Returns the alpha/beta tables (i.e. the factor messages) using
        belief propagation (which is equivalent to forward-backward in HMMs).
        """
        
        #Filling alpha table
        targetLen = np.shape(self.alpha)[1]
        for k in range(1, targetLen):
            for i in range(0, self.n_classes):
                #log-sum-exp-- modify it to handle exp. overflow
                alpha_tmp = np.zeros((self.n_classes))
                for j in range(0, self.n_classes):
                    alpha_tmp[j] = (self.target_unary_log_factors[k-1][j] + 
                                  self.lateral_weights[j][i] + self.alpha[j][k-1])
                maxi = np.max(alpha_tmp)
                self.alpha[i][k] = maxi + np.log(np.sum(np.exp(alpha_tmp - maxi)))
        
        #Filling beta table
        for k in range(targetLen-2, -1, -1):
            for i in range(0, self.n_classes):
                beta_tmp = np.zeros(self.n_classes)
                #log-sum-exp-- modify it to handle exp. overflow
                for j in range(0, self.n_classes):
                    beta_tmp[j] = (self.target_unary_log_factors[k+1][j] + 
                                  self.lateral_weights[i][j] + self.beta[j][k+1])
                maxi = np.max(beta_tmp)
                self.beta[i][k] = maxi + np.log(np.sum(np.exp(beta_tmp - maxi)))
        
        ### Now both alpha and beta tables are filled
        ###
        ###
    
    def training_loss(self,target,target_unary_log_factors,alpha,beta):
        """
        Computation of the loss:
        - returns the regularized negative log-likelihood loss associated with the
          given the true target, the unary log factors of the target space and alpha/beta tables
        """
        targetLen = len(target)
        #log-sum-exp operation
        partition_tmp = np.zeros((self.n_classes))
        for j in range(0, self.n_classes):
              partition_tmp[j] = (target_unary_log_factors[targetLen-1][j] + 
                                           alpha[j][targetLen-1])
        maxi = np.max(partition_tmp)
        log_partition_func = maxi + np.log(np.sum(np.exp(partition_tmp - maxi)))
        
        log_numerator = 0
        for k in range(0, targetLen):
            y_k = target[k] #class label between 0 and n_classes
            y_k_1 = 0
            paired_activation = 0
            if(k < (targetLen-1)):
                y_k_1 = target[k+1]
                paired_activation = self.lateral_weights[y_k][y_k_1]
            log_numerator += (target_unary_log_factors[k][y_k] + paired_activation)
        
        log_negative_likelihood = -(log_numerator - log_partition_func)
        
        weight_squares = np.square(self.weights[0]) + np.square(self.weights[1]) + np.square(self.weights[2])
        lateral_wt_square = np.square(self.lateral_weights)
        
        absolute_weights = ( np.absolute(self.weights[0]) + np.absolute(self.weights[1]) + 
                             np.absolute(self.weights[2]))
        abs_lateral_wts = np.absolute(self.lateral_weights)
        regularized_training_loss = (log_negative_likelihood + self.L2 * (np.sum(weight_squares) + 
                                     np.sum(lateral_wt_square)) + self.L1 * (np.sum(absolute_weights) + 
                                                                             np.sum(abs_lateral_wts)))
        
        return regularized_training_loss


    def bprop(self,input,target):
        """
        Backpropagation:
        - fills in the CRF gradients of the weights, lateral weights and bias 
          in self.grad_weights, self.grad_lateral_weights and self.grad_bias
        - returns nothing
        Argument ``input`` is a Numpy 2D array where the number of
        rows if the sequence size and the number of columns is the
        input size. 
        Argument ``target`` is a Numpy 1D array of integers between 
        0 and nb. of classe - 1. Its size is the same as the number of
        rows of argument ``input``.
        """
        targetLen = len(target)
        marginal_probas = np.zeros((targetLen, self.n_classes))
        
        #computing marginals for all possible outcomes(classes)
        #k = 0 means y1, k=1 means y2,..so on
        for k in range(0, targetLen):
            sum = 0
            tmp_arr = np.zeros((self.n_classes))
            for i in range(0, self.n_classes):
                tmp_arr[i] = (self.target_unary_log_factors[k][i] +
                                               self.alpha[i][k] + self.beta[i][k])
                #marginal_probas[k][i] = tmp_arr[i]
            #log-sum-exp operation
            maxi = np.max(tmp_arr)
            log_sum = maxi + np.log(np.sum(np.exp(tmp_arr - maxi)))
            log_probas = tmp_arr - log_sum
            #normalizing marginal probabilities
            for i in range(0, self.n_classes):
                marginal_probas[k][i] = np.exp(log_probas[i])
        
        #computing p(y_k |X) - e(y_k) as required in gradients of weights
        #prob_minus_target = np.zeros((targetLen, self.n_classes))
        prob_minus_target = marginal_probas.copy()
        for k in range(0, targetLen):
            y_k = target[k] #a label belonging to [0, n_classes)
            prob_minus_target[k][y_k] -= 1
        
        #1. Calculating self.grad_bias
        for k in range(0, targetLen):
            self.grad_bias += prob_minus_target[k]
        
        #2. Calculating self.grad_weights
        self.grad_weights[0] = 0
        self.grad_weights[1] = 0
        self.grad_weights[2] = 0
        for k in range(0, targetLen):
            #Weight Gradients for left inputs
            if (k > 0):
                self.grad_weights[0] += np.outer(input[k-1],prob_minus_target[k])
                
            #Weight Gradients for central inputs
            self.grad_weights[1] += np.outer(input[k], prob_minus_target[k])
            
            #Weight Gradients for right inputs
            if (k < (targetLen-1)):
                self.grad_weights[2] += np.outer(input[k+1], prob_minus_target[k])
        
        #3. Calculating self.grad_lateral_weights
        for k in range(0, targetLen-1):
            #Allocate space for pairwise marginal probabilities
            pairwise_marginal_probas = np.zeros((self.n_classes, self.n_classes))
            tmp_arr  = np.zeros((self.n_classes, self.n_classes))
            # i --> y_k ; j --> y_kplus1
            for i in range(0, self.n_classes):
                for j in range(0, self.n_classes):
                    tmp_arr[i][j] = (self.target_unary_log_factors[k][i] + 
                                                            self.lateral_weights[i][j] + 
                                                            self.target_unary_log_factors[k+1][j] +
                                                            self.alpha[i][k] + self.beta[j][k+1])
                    #pairwise_marginal_probas[i][j] = tmp_arr[i][j]
            
            #log-sum-exp operation
            maxi = np.max(tmp_arr)
            log_sum = maxi + np.log(np.sum(np.exp(tmp_arr - maxi)))
            log_probas = tmp_arr - log_sum
            for i in range(0, self.n_classes):
                for j in range(0, self.n_classes):
                    pairwise_marginal_probas[i][j] = np.exp(log_probas[i][j])
            
            
            #subtracting 1 from above matrix for true target labels y_k, y_kplus1
            y_k = target[k]
            y_kplus1 = target[k+1]
            pairwise_marginal_probas[y_k][y_kplus1] -= 1
            
            #Adding pairwise probabilities minus true targets to self.grad_lateral_weights
            self.grad_lateral_weights += pairwise_marginal_probas
            
        

    def update(self):
        """
        Stochastic gradient update:
        - performs a gradient step update of the CRF parameters self.weights,
          self.lateral_weights and self.bias, using the gradients in 
          self.grad_weights, self.grad_lateral_weights and self.grad_bias
        """
        ##Both L2 and L1 regularization have been used
        self.bias = self.bias - self.lr * self.grad_bias
        self.weights[0] = ( self.weights[0] + self.lr * (-self.grad_weights[0] - self.L2 * 2 * self.weights[0] - 
                            self.L1 * np.sign(self.weights[0])))
        self.weights[1] = ( self.weights[1] + self.lr * (-self.grad_weights[1] - self.L2 * 2 * self.weights[1] - 
                            self.L1 * np.sign(self.weights[1])))
        self.weights[2] = ( self.weights[2] + self.lr * (-self.grad_weights[2] - self.L2 * 2 * self.weights[2] - 
                            self.L1 * np.sign(self.weights[2])))
        self.lateral_weights = ( self.lateral_weights + self.lr * (-self.grad_lateral_weights - 
                                 self.L2 * 2 * self.lateral_weights - self.L1 * np.sign(self.lateral_weights)
                                ))
        
        self.lr -= self.dc
    
    def predict(self, input):
        #Viterbi Decoding Algorithm
        #Assuming input is 2D Numpy Array
        targetLen = np.shape(input)[0]
        self.alpha_star = np.zeros((self.n_classes, targetLen))
        self.alpha_decoder = np.zeros((self.n_classes, targetLen))
        
        for k in range(1, targetLen):
            for j in range(0, self.n_classes):
                for i in range(0, self.n_classes):
                    val = ( self.target_unary_log_factors[k-1][i] + self.lateral_weights[i][j] + 
                            self.alpha_star[i][k-1])
                    if val > self.alpha_star[j][k]:
                        self.alpha_star[j][k] = val
                        self.alpha_decoder[j][k] = i
        
        max_probability = 0
        max_prob_yK = 0
        for i in range(0, self.n_classes):
            val = (self.target_unary_log_factors[targetLen-1][i] + self.alpha_star[i][targetLen-1])
            if val > max_probability:
                max_probability = val
                max_prob_yK = i
        
        targets = np.zeros((targetLen))
        targets[targetLen-1] = max_prob_yK
        for k in range(targetLen-2, -1, -1):
            targets[k] = self.alpha_decoder[targets[k+1]][k+1]
        
        return targets

    def use(self,dataset):
        """
        Computes and returns the outputs of the Learner for
        ``dataset``:
        - the outputs should be a list of size ``len(dataset)``, containing
          a Numpy 1D array that gives the class prediction for each position
          in the sequence, for each input sequence in ``dataset``
        Argument ``dataset`` is an MLProblem object.
        """
        outputs = []
        for input, target in dataset:
            #Assuming dataset[i] is a tuple input, target
            test_input = np.array(input) #2-D Numpy array
            test_target = np.array(target) #1-D Numpy array
            #test_target is not used anywhere in fprop except training loss calculation
            self.fprop(test_input, test_target)
            targets = self.predict(test_input)
            outputs.append(targets)
            
        return outputs
        
    def test(self,dataset):
        """
        Computes and returns the outputs of the Learner as well as the errors of the
        CRF for ``dataset``:
        - the errors should be a list of size ``len(dataset)``, containing
          a pair ``(classif_errors,nll)`` for each examples in ``dataset``, where 
            - ``classif_errors`` is a Numpy 1D array that gives the class prediction error 
              (0/1) at each position in the sequence
            - ``nll`` is a positive float giving the regularized negative log-likelihood of the target given
              the input sequence
        Argument ``dataset`` is an MLProblem object.
        """
        outputs = self.use(dataset)
        errors = []

        i = 0
        for org_input, org_target in dataset:
            classif_errors = np.zeros((len(org_target)))
            pred_target = outputs[i]
            
            #Remove below 5 lines
#             if self.n_epochs == 6:
#                 org_target_int = org_target.astype(int) + 97
#                 pred_target_int = pred_target.astype(int) + 97
#                 print 'org_target: ', ''.join(chr(i) for i in org_target_int)
#                 print '\npred_target: ', ''.join(chr(i) for i in pred_target_int)
#                 print '\n\n'

            for ind in range(0, len(org_target)):
                if(org_target[ind] != pred_target[ind]):
                    classif_errors[ind] = 1
            i += 1
            nll = self.fprop(org_input, pred_target)
            tmp = (classif_errors, nll)
            errors.append(tmp)
            
        return outputs, errors
 
    def verify_gradients(self):
        """
        Verifies the implementation of the fprop and bprop methods
        using a comparison with a finite difference approximation of
        the gradients.
        """
        
        print 'WARNING: calling verify_gradients reinitializes the learner'
  
        rng = np.random.mtrand.RandomState(1234)
  
        self.initialize(10,3)
        example = (rng.rand(4,10),np.array([0,1,1,2]))
        input,target = example
        epsilon=1e-6
        self.lr = 0.1
        self.decrease_constant = 0

        self.weights = [0.01*rng.rand(self.input_size,self.n_classes),
                        0.01*rng.rand(self.input_size,self.n_classes),
                        0.01*rng.rand(self.input_size,self.n_classes)]
        self.bias = 0.01*rng.rand(self.n_classes)
        self.lateral_weights = 0.01*rng.rand(self.n_classes,self.n_classes)
        
        self.fprop(input,target)
        self.bprop(input,target) # compute gradients

        import copy
        emp_grad_weights = copy.deepcopy(self.weights)
  
        for h in range(len(self.weights)):
            for i in range(self.weights[h].shape[0]):
                for j in range(self.weights[h].shape[1]):
                    self.weights[h][i,j] += epsilon
                    a = self.fprop(input,target)
                    self.weights[h][i,j] -= epsilon
                    
                    self.weights[h][i,j] -= epsilon
                    b = self.fprop(input,target)
                    self.weights[h][i,j] += epsilon
                    
                    emp_grad_weights[h][i,j] = (a-b)/(2.*epsilon)


        print 'grad_weights[-1] diff.:',np.sum(np.abs(self.grad_weights[-1].ravel()-emp_grad_weights[-1].ravel()))/self.weights[-1].ravel().shape[0]
        print 'grad_weights[0] diff.:',np.sum(np.abs(self.grad_weights[0].ravel()-emp_grad_weights[0].ravel()))/self.weights[0].ravel().shape[0]
        print 'grad_weights[1] diff.:',np.sum(np.abs(self.grad_weights[1].ravel()-emp_grad_weights[1].ravel()))/self.weights[1].ravel().shape[0]
  
        emp_grad_lateral_weights = copy.deepcopy(self.lateral_weights)
  
        for i in range(self.lateral_weights.shape[0]):
            for j in range(self.lateral_weights.shape[1]):
                self.lateral_weights[i,j] += epsilon
                a = self.fprop(input,target)
                self.lateral_weights[i,j] -= epsilon

                self.lateral_weights[i,j] -= epsilon
                b = self.fprop(input,target)
                self.lateral_weights[i,j] += epsilon
                
                emp_grad_lateral_weights[i,j] = (a-b)/(2.*epsilon)


        print 'grad_lateral_weights diff.:',np.sum(np.abs(self.grad_lateral_weights.ravel()-emp_grad_lateral_weights.ravel()))/self.lateral_weights.ravel().shape[0]

        emp_grad_bias = copy.deepcopy(self.bias)
        for i in range(self.bias.shape[0]):
            self.bias[i] += epsilon
            a = self.fprop(input,target)
            self.bias[i] -= epsilon
            
            self.bias[i] -= epsilon
            b = self.fprop(input,target)
            self.bias[i] += epsilon
            
            emp_grad_bias[i] = (a-b)/(2.*epsilon)

        print 'grad_bias diff.:',np.sum(np.abs(self.grad_bias.ravel()-emp_grad_bias.ravel()))/self.bias.ravel().shape[0]
