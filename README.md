This was my project for the final week of the ARENA program. 

The aim was to create a small toy model on which to experiment, in order to see how RL post-training affects the behaviour of a model.
The dataset on which the base model was trained was the follows:

1/3rd "A110110"  
1/3rd "B001001"  
1/3rd "C------", with - a coinflip of 0 or 1  

The model is then trained with RL to maximise the number of 1s. 
Due to the KL divergence in the reward function, this model continues to output the string "110110" after seeing character "A" and "001001" after character B.
It therefore chooses "A" more often than "B". However, due to the inherent randomness in "C", the KL divergence doesn't give a strong penalty for an all 1s output. Under most parameter choices, "C" is therefore the most chosen character. 

This is a simple demonstration of RL pushing models towards unpredictable areas of the search space, 
and subsequent tests on other reward functions with similar results make this a simple demonstration of Instrumental Convergence on a 220 parameter model.
