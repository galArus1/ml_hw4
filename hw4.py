import numpy as np
# Auxilary functions
def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    m = len(X)
    X = np.c_[np.ones(m), X]

    return X
def compute_sigmoid(v):
    """
    Computes the sigmoid function for the input vector v.

    Input:
    - v: Input vector (m instances over n features).

    Returns:
    - Sigmoid of v (m instances over n features).
    """
    return 1 / (1 + np.exp(-v))

def compute_cost(X, y, theta):
    """
    Computes the cost function for logistic regression.

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    theta : array-like, shape = [n_features]
      Model parameters.

    Returns
    -------
    J : float
      The computed cost.
    """
    m = X.shape[0]
    h = compute_sigmoid(X @ theta)
    epsilon = 1e-15  # to prevent log(0)
    J = (1 / m) * (-y @ np.log(h + epsilon) - (1 - y) @ np.log(1 - h + epsilon))
    return J
        

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Data preprocessing
        X = apply_bias_trick(X)
        m, n = X.shape
        self.theta = np.random.random(n)  # Initialize theta with random values
        # Learning process
        theta, j_history, theta_history = self.gradinet_descent(X, y, self.theta)
        # Store the final theta and cost history
        self.theta = theta
        self.Js = j_history
        self.thetas.append(theta_history)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = apply_bias_trick(X)  # Apply bias trick to the input data
        preds = np.round(compute_sigmoid( X @ self.theta)).astype(int)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds
  
    def gradinet_descent(self, X, y, theta):
        theta = theta.copy() # deep copy in order to not change the original theta
        J_history = []
        theta_history = []
        delta_J  = float('inf')  # Initialize J_delta to a large value
        m = X.shape[0]

        for i in range(self.n_iter):
            h = compute_sigmoid(X @ theta)
            theta = theta - (self.eta / m) * (X.T @ (h - y))
            J_i = compute_cost(X, y, theta)
            J_history.append(J_i)
            theta_history.append(theta.copy())  # Store the current theta

            if len(J_history) > 1:
                delta_J = abs(J_history[i] - J_history[i-1])
                
                if delta_J < self.eps:
                    break
        
        return theta, J_history, theta_history

            
         


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Shuffle the data
    indices = np.arange(0, len(X), 1)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
   
    # Create folds
    fold_size = len(X) // folds
    accuracies = []
    
    for i in range(folds):
        # Create the start and end indices for the current fold
        start_i = i * fold_size
        end_i = start_i + fold_size if i < folds-1 else len(X)

        # Split the data into training and validation sets
        X_train = np.concatenate((X[:start_i], X[end_i:]), axis=0)
        y_train = np.concatenate((y[:start_i], y[end_i:]), axis=0)
        X_val = X[start_i:end_i]
        y_val = y[start_i:end_i]

        # Fit the model on the training set
        algo.fit(X_train, y_train)

        # Predict on the validation set
        preds = algo.predict(X_val)

        # Calculate accuracy for this fold
        accuracy = np.mean(preds == y_val)
        accuracies.append(accuracy)
    
    # Calculate the average accuracy across all folds
    cv_accuracy = np.mean(accuracies)   
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p = np.exp( -(((data - mu) / sigma) ** 2) / 2) / (sigma * np.sqrt(2 * np.pi))
  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p


def norm_pdf2(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    sigma = np.maximum(sigma, 1e-10)  # prevent division by zero
    coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((data - mu) / sigma) ** 2
    p = coeff * np.exp(exponent)
  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Initialize weights uniformly
        self.weights = np.ones(self.k) / self.k
        
        # Initialize mus uniformly
        self.mus = np.random.uniform(np.min(data), np.max(data), self.k)
        
        # Initialize sigmas randomly and avoid zero division
        self.sigmas = np.random.random(self.k) + 1e-2  
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihoods = np.array([
            self.weights[j] * norm_pdf(data, self.mus[j], self.sigmas[j])
            for j in range(self.k)
        ])
        # Calculate and Normalize responsibilities
        self.responsibilities = likelihoods / (np.sum(likelihoods, axis=0) + 1e-10)  # Avoid division by zero
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        n = data.shape[0]
        gamma_sum = np.sum(self.responsibilities, axis=1)
        # update parameters
        self.weights = gamma_sum / n
        self.mus = np.sum(self.responsibilities * data, axis=1) / gamma_sum
        self.sigmas = np.sqrt(
            np.sum(
                self.responsibilities * (data - self.mus[:, np.newaxis])**2, axis=1
            ) / gamma_sum
        )
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)
        self.costs = []
        prev_cost = float('inf')

        for _ in range(self.n_iter):
            # E step
            self.expectation(data)
            # M step
            self.maximization(data)

            # Calculate the cost (log likelihood)
            likelihoods = np.array([
                self.weights[j] * norm_pdf(data, self.mus[j], self.sigmas[j])
                for j in range(self.k)
            ])

            log_likelihood = np.sum(np.log(np.sum(likelihoods, axis=0) + 1e-10))
            cost = -log_likelihood
            self.costs.append(cost)

            if abs(prev_cost - cost) < self.eps:
                break
            prev_cost = cost

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    k = len(weights)
    pdf = np.zeros_like(data.flatten(), dtype=float)
    for j in range(k):
        component_pdf = norm_pdf(data, mus[j], sigmas[j])
        pdf += weights[j] * component_pdf # weighted sum of the component PDFs
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.models = None  # EM models:  dict[class -> list of EM objects (per feature)]

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.models = {}
        self.prior = {}
        classes = np.unique(y)

        for c in classes:
            X_c = X[y == c]  # Get all instances of class c
            self.prior[c] = len(X_c) / len(y)  # Calculate prior probability for class c
            self.models[c] = []
            for feature in range(X.shape[1]):
                # Create an EM model for each feature
                em = EM(k=self.k, random_state=self.random_state)
                em.fit(X_c[:, feature].reshape(-1, 1))  # Fit on one feature column
                self.models[c].append(em)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        preds = []
        for i in range(X.shape[0]):
            x_i = X[i]
            class_probs = {}

            for c in self.models:
                # set the prior as the initial probability for class c
                prob = self.prior[c]
                # Multiply likelihoods for each feature assuming conditional independence/Naive part
                for j in range(len(x_i)):
                    em = self.models[c][j]
                    weights, mus, sigmas = em.get_dist_params()
                    p = gmm_pdf(x_i[j], weights, mus, sigmas)[0]  # Get the PDF value for the feature
                    prob *= p
                class_probs[c] = prob
        
            # Find the class with the maximum probability
            pred_class = max(class_probs, key=class_probs.get)
            preds.append(pred_class)
        
        preds = np.array(preds)  # Keep on consistency
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Train Logistic Regression model
    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    y_train_preds_lor = lor.predict(x_train)
    y_test_preds_lor = lor.predict(x_test)

    # Train Naive Bayes Gaussian model
    bayes = NaiveBayesGaussian(k=k)
    bayes.fit(x_train, y_train)
    y_train_preds_bayes = bayes.predict(x_train)
    y_test_preds_bayes = bayes.predict(x_test)
    # Calculate accuracies
    lor_train_acc = np.mean(y_train_preds_lor == y_train)
    lor_test_acc = np.mean(y_test_preds_lor == y_test)
    bayes_train_acc = np.mean(y_train_preds_bayes == y_train)
    bayes_test_acc = np.mean(y_test_preds_bayes == y_test)

    # Adding 'models' to the return dict for visualization purposes
    models = {
        'lor': lor,
        'bayes': bayes
    }
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc,
            'models': models
            }

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    random_state = 1991
    np.random.seed(random_state)
    n_samples = 500

    # Dataset A - Naive Bayes will outperform Logistic Regression
    # Features are conditionally independent given the class (diagonal covariance)
    mean_a0 = [0, 0, 0]
    mean_a1 = [1, 1, 1]
    cov_a = np.diag([1, 1,1]) # Independent features
    
    X_a0 = multivariate_normal.rvs(mean=mean_a0, cov=cov_a, size=n_samples)
    X_a1 = multivariate_normal.rvs(mean=mean_a1, cov=cov_a, size=n_samples)
    dataset_a_features = np.vstack((X_a0, X_a1))
    dataset_a_labels = np.array([0] * n_samples + [1] * n_samples) # Labels for dataset A

    # Dataset B - Logistic Regression will outperform Naive Bayes
    # Features are strongly correlated, breaking NB's assumption
    mean_b0 = [0, 0, 0]
    mean_b1 = [1, 1, 1]
    cov_b = np.array([[1, 0.9, 0.9],
                      [0.9, 1, 0.9],
                      [0.9, 0.9, 1]])  # correlated features

    X_b0 = multivariate_normal.rvs(mean=mean_b0, cov=cov_b, size=n_samples)
    X_b1 = multivariate_normal.rvs(mean=mean_b1, cov=cov_b, size=n_samples)
    dataset_b_features = np.vstack([X_b0, X_b1])
    dataset_b_labels = np.array([0] * n_samples + [1] * n_samples)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }