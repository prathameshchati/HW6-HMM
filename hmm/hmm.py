import numpy as np
import warnings
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    # helper functions
    


    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p

        self.viterbi_table=None # for later

        # check matrix dimensions of emission matrix; should have same number of rows as hidden states and same number of columns as observed states, otherwise raise error
        if not (emission_p.shape==(len(hidden_states), len(observation_states))):
            raise ValueError("The dimensions of the emission matrix are not correct. Please make sure they the number of rows and columns match the number of hidden and observed states, respectively.")
        
        # check that dimensions of the transition matrix are square and equal to number of hidden states
        if not (transition_p.shape==(len(hidden_states), len(hidden_states))):
            raise ValueError("The dimensions of the transition matrix are not correct. Please make sure they the number of rows and columns match the number of hidden states.")

        # check prior probabilities equal number of hidden states
        if not (prior_p.shape==(len(hidden_states),)):
            raise ValueError("The number of prior probabilities given is not equal to the number of hidden states.")
        
        # check that the sum of transition probabilities across columns (so sum of each row), is equal to 1
        if not np.array_equal(transition_p.sum(axis=1),np.ones(transition_p.sum(axis=1).shape)):
            raise ValueError("The sum of transitions across the columns are not equal to 1, please make sure you are using probabilities that sum to 1.")
        
        # check that the sum of emission probabilities across columns (so sum of each row), is equal to 1
        if not np.array_equal(emission_p.sum(axis=1),np.ones(emission_p.sum(axis=1).shape)):
            raise ValueError("The sum of emissions across the columns are not equal to 1, please make sure you are using probabilities that sum to 1.")


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        

        # raise warning if there exists a transition probability with 0 emission
        if not np.all(self.transition_p!=0):
            warnings.warn("Note, transition probabilities contain zeroes, this may throw off the predicted sequences if included by mistake.")

        
        # Step 1. Initialize variables
        self.input_observation_states=input_observation_states
        alpha_vector=np.multiply(self.prior_p, self.emission_p[:,self.observation_states_dict[self.input_observation_states[0]]]) # initialize alpha as the product between each hidden state's prior and initial observed conditional probability

        # Step 2. Calculate probabilities

        # loop through remaining observations and iteratively update alpha
        for o in range(1, len(self.input_observation_states)):
            alpha_vector=np.multiply(self.emission_p[:,self.observation_states_dict[self.input_observation_states[o]]], np.dot(alpha_vector, self.transition_p))

        alpha=alpha_vector.sum()

        # Step 3. Return final probability 
        return alpha
        

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        

        # raise warning if there exists a transition probability with 0 emission
        if not np.all(self.transition_p!=0):
            warnings.warn("Note, transition probabilities contain zeroes, this may throw off the predicted sequences if included by mistake.")

        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        # viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        viterbi_table=np.zeros((len(self.hidden_states), len(decode_observation_states))) # reversed dimensions for matrix formatting

        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))         
        
    
        # Step 2. Calculate Probabilities
        # general reference: https://en.wikipedia.org/wiki/Viterbi_algorithm

        pointers=np.zeros((len(self.hidden_states), len(decode_observation_states)))
        viterbi_table=np.zeros((len(self.hidden_states), len(decode_observation_states)))
        viterbi_table[:,0]=np.multiply(self.prior_p, self.emission_p[:,self.observation_states_dict[decode_observation_states[0]]])

        # iterate through all observations (start at 1 because we computed index 0 for the first observation with the prior probabilities of the hidden state)
        for o in range(1, len(decode_observation_states)):
            # here, the nth column corresponds to the marginal alpha values for the nth hidden state (the sum of these alpha values in the nth column gives you the alpha value for that hidden state)
            marg_alpha_table=np.multiply(np.transpose(np.multiply(viterbi_table[:,o-1], np.transpose(self.transition_p))), self.emission_p[:,self.observation_states_dict[decode_observation_states[o]]])
            viterbi_table[:,o]=np.max(marg_alpha_table, axis=0)
            pointers[:,o]=np.argmax(marg_alpha_table, axis=0)

            
        # Step 3. Traceback 
        k=np.argmax([viterbi_table[k, len(decode_observation_states)-1] for k in range(len(self.hidden_states))])
        best_path=[]
        for o in range(len(decode_observation_states)-1,-1,-1):
            best_path.append(self.hidden_states[k])
            k=int(pointers[k,o])

        best_path.reverse()
        best_path=np.array(best_path)

        # add viterbi table to self for testing later
        self.viterbi_table=viterbi_table

        # Step 4. Return best hidden state sequence 
        return best_path
        