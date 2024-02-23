import pytest
from hmm import HiddenMarkovModel
from hmmlearn.hmm import CategoricalHMM
import numpy as np
import warnings


# categorical to numerical sequence 
def encode_seq(seq, encodings):
    return np.array([encodings.get(i, i) for i in seq])

# reverse dictionary 
def reverse_dictionary(d):
    return {v: k for k, v in d.items()}


def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    # check edge cases in separate test below

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    # get vars
    prior_p=mini_hmm["prior_p"]
    transition_p=mini_hmm["transition_p"]
    emission_p=mini_hmm["emission_p"]
    hidden_states=mini_hmm["hidden_states"]
    observation_states=mini_hmm["observation_states"]

    # create cat to num dict
    hidden_states_dict=reverse_dictionary(dict(enumerate(hidden_states)))

    # initialize hmm model
    hmm_model=HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    # initialize hmmlearn model
    hmmlearn_model=CategoricalHMM(n_components=len(prior_p))  
    hmmlearn_model.startprob_=prior_p
    hmmlearn_model.transmat_=transition_p
    hmmlearn_model.emissionprob_=emission_p

    # get sequences
    observation_state_sequence=mini_input["observation_state_sequence"]
    best_hidden_state_sequence=mini_input["best_hidden_state_sequence"]

    # encode sequences for testing with hmlearn (sklearn) catagorical hmm
    observation_state_sequence_encoded=encode_seq(observation_state_sequence, hmm_model.observation_states_dict)
    best_hidden_state_sequence_encoded=encode_seq(best_hidden_state_sequence, hidden_states_dict)

    # run forward algorithm with our implementation and hmmlearn version 
    hmm_model_forward_alpha=hmm_model.forward(observation_state_sequence)
    hmmlearn_model_forward_alpha=np.exp(hmmlearn_model.score(observation_state_sequence_encoded.reshape(-1, 1))) # hmmlearn returns log probability using the forward-backward algorithm (we exponentiate to get probability)

    # check that our likelihoods for the two methods are roughly the same within some error bounds
    assert (hmm_model_forward_alpha-hmmlearn_model_forward_alpha) < 0.000000001
    
    # run viterbi algorithm
    hmm_model_best_path=hmm_model.viterbi(observation_state_sequence)
    hmm_model_best_path_encoded=encode_seq(hmm_model_best_path, hidden_states_dict) # encode our predicted best sequence for checking with hmmlearn
    hmm_model_viterbi_table_last_prob=hmm_model.viterbi_table[len(hidden_states)-1, len(observation_state_sequence)-1]

    # the decode function gives you the last carry through probability (logged) in the viterbi table along with the encoded best path
    hmmlearn_model_viterbi_table_last_prob, hmmlearn_model_best_path_encoded=hmmlearn_model.decode(observation_state_sequence_encoded.reshape(-1, 1), algorithm='viterbi')
    
    # check that we have the correct number of output predicted hidden states
    assert len(hmm_model_best_path)==len(observation_state_sequence)
    assert len(hmm_model_best_path)==len(best_hidden_state_sequence)

    # check that our algorithm matches the provided best sequence
    assert np.array_equal(hmm_model_best_path,best_hidden_state_sequence)

    # assert the same best paths are returned via the two algorithms
    assert np.array_equal(hmm_model_best_path_encoded, hmmlearn_model_best_path_encoded)
    
    # check that the last probabilties are roughly the same (these values are carried through, so it should indicate that the process was run correctly)
    assert (hmm_model_viterbi_table_last_prob-np.exp(hmmlearn_model_viterbi_table_last_prob)) < 0.000000001
    
    # pass


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    # test the same things as above

    # load in dataset
    mw_hmm=np.load('./data/full_weather_hmm.npz')
    mw_seq=np.load('./data/full_weather_sequences.npz')

    # get vars
    prior_p=mw_hmm["prior_p"]
    transition_p=mw_hmm["transition_p"]
    emission_p=mw_hmm["emission_p"]
    hidden_states=mw_hmm["hidden_states"]
    observation_states=mw_hmm["observation_states"]

    # create cat to num dict
    hidden_states_dict=reverse_dictionary(dict(enumerate(hidden_states)))

    # initialize hmm model
    hmm_model=HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    # initialize hmmlearn model
    hmmlearn_model=CategoricalHMM(n_components=len(prior_p))  
    hmmlearn_model.startprob_=prior_p
    hmmlearn_model.transmat_=transition_p
    hmmlearn_model.emissionprob_=emission_p

    # get sequences
    observation_state_sequence=mw_seq["observation_state_sequence"]
    best_hidden_state_sequence=mw_seq["best_hidden_state_sequence"]

    # encode sequences for testing with hmlearn (sklearn) catagorical hmm
    observation_state_sequence_encoded=encode_seq(observation_state_sequence, hmm_model.observation_states_dict)
    best_hidden_state_sequence_encoded=encode_seq(best_hidden_state_sequence, hidden_states_dict)

    # run forward algorithm with our implementation and hmmlearn version 
    hmm_model_forward_alpha=hmm_model.forward(observation_state_sequence)
    hmmlearn_model_forward_alpha=np.exp(hmmlearn_model.score(observation_state_sequence_encoded.reshape(-1, 1))) # hmmlearn returns log probability using the forward-backward algorithm (we exponentiate to get probability)

    # check that our likelihoods for the two methods are roughly the same within some error bounds
    assert (hmm_model_forward_alpha-hmmlearn_model_forward_alpha) < 0.000000001
    
    # run viterbi algorithm
    hmm_model_best_path=hmm_model.viterbi(observation_state_sequence)
    hmm_model_best_path_encoded=encode_seq(hmm_model_best_path, hidden_states_dict) # encode our predicted best sequence for checking with hmmlearn
    hmm_model_viterbi_table_last_prob=hmm_model.viterbi_table[len(hidden_states)-1, len(observation_state_sequence)-1]

    # the decode function gives you the last carry through probability (logged) in the viterbi table along with the encoded best path
    hmmlearn_model_viterbi_table_last_prob, hmmlearn_model_best_path_encoded=hmmlearn_model.decode(observation_state_sequence_encoded.reshape(-1, 1), algorithm='viterbi')
    
    # check that we have the correct number of output predicted hidden states
    assert len(hmm_model_best_path)==len(observation_state_sequence)
    assert len(hmm_model_best_path)==len(best_hidden_state_sequence)

    # check that our algorithm matches the provided best sequence
    assert np.array_equal(hmm_model_best_path,best_hidden_state_sequence)

    # assert the same best paths are returned via the two algorithms
    assert np.array_equal(hmm_model_best_path_encoded, hmmlearn_model_best_path_encoded)
    
    # check that the last probabilties are roughly the same (these values are carried through, so it should indicate that the process was run correctly)
    assert (hmm_model_viterbi_table_last_prob-np.exp(hmmlearn_model_viterbi_table_last_prob)) < 0.000000001
    
    # pass

def test_edge_cases():
    
    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    # get vars
    prior_p=mini_hmm["prior_p"]
    transition_p=mini_hmm["transition_p"]
    emission_p=mini_hmm["emission_p"]
    hidden_states=mini_hmm["hidden_states"]
    observation_states=mini_hmm["observation_states"]

    # get sequences
    observation_state_sequence=mini_input["observation_state_sequence"]
    best_hidden_state_sequence=mini_input["best_hidden_state_sequence"]

    # create cat to num dict
    hidden_states_dict=reverse_dictionary(dict(enumerate(hidden_states)))

    # initialize hmm model and check that error is raised if any of the dimension constraints are violated
    with pytest.raises(ValueError) as hmm_model:
        HiddenMarkovModel(observation_states, hidden_states, np.array([0.6, 0.2, 0.2]), transition_p, emission_p)
    assert hmm_model.type==ValueError

    with pytest.raises(ValueError) as hmm_model:
        HiddenMarkovModel(observation_states, hidden_states, prior_p, np.array([[0.5, 0.25, 0.25],[0.3 , 0.7 ]]), emission_p)
    assert hmm_model.type==ValueError

    with pytest.raises(ValueError) as hmm_model:
        HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, np.array([[0.5, 0.25, 0.25],[0.3 , 0.7 ]]))
    assert hmm_model.type==ValueError

    # check that error raised when transition or emission probabilities don't sum to 1
    with pytest.raises(ValueError) as hmm_model:
        HiddenMarkovModel(observation_states, hidden_states, prior_p, np.array([[0.5, 0.4],[0.3 , 0.7 ]]), emission_p)
    assert hmm_model.type==ValueError

    with pytest.raises(ValueError) as hmm_model:
        HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, np.array([[0.5, 0.4],[0.3 , 0.7 ]]))
    assert hmm_model.type==ValueError

    # check that warning is thrown with zero included in transition matrix but still runs
    hmm_model=HiddenMarkovModel(observation_states, hidden_states, prior_p, np.array([[1, 0],[0.3 , 0.7 ]]), emission_p)
    with pytest.warns(UserWarning):
        hmm_model_forward_alpha=hmm_model.forward(observation_state_sequence)



    # pass











