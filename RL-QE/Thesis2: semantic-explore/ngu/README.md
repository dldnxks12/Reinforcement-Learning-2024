#### Never Give Up (NGU)

---

- Brief summary of NGU


            * 1 line summary

                they propose the NGU intrinsic reward, that does not 'vanish' over time!

                * A core contribution :

                a method for learning policies that can maintain exploration throughout the training process. 
                ➔ In Depth-First-Search (DFS) mannar



            * preliminaries * 

            1) Problems in exploration and limitations   
            
            Methods that guarantee finding an optimal policy require the number of visits to each state-action parir to approach infinity.
            ➔ ensuring that all state-action pairs are encountered infinitely often is the general problem of maintaining exploration.

            The simplest approach is to consider stochastic policies with a non-zero prob. of selecting all actions in each state.
            ➔ e.g. ε-greedy or Boltzmann exploration

            These methods will eventually learn the optimal policy in 'tabular-setting'.
            ➔ but, they are very inefficient and the steps they require grow exponentially with the size of the state space.

            Despite these disadvantages, they perform well in dense reward settings.
            ➔ but, in sparse reward setting, they fail to learn, as it is hard to find the very few rewarding states.
            (temporally-extended exploration or deep exploration is crucial)

                
            2) overcome the issues in sparse setting : intrinsic reward

            To demonstrate performance even in sparse reward settings, 
            researchers suggests to provide 'intrinsic rewards' to agent to drive exploration.

            * Intrinsic reward setting (1)
               make rewards as proportional to quantity of how difference the current state is from those already visited.
                ➔ as the agent explores the env. and becomes familiar with it, the exploration bonus disapears
                and learning is only driven by extrinsic rewards. 
                    ➔ very sensible idea as the goal is to maximizse the expected sum of extrinsic rewards.

                * limitation of (1) 
                    after the novelty of a state has vanished, the agent is not encouraged to visit it again.


            * Intrinsic reward setting (2)
                make predictive forward models and use the prediction error as the intrinsic motivations (e.g. icm)
                    
                * limitation of (2)
                    explicitly building predictive models, particulary from observations, is expensive and error prone.
                    In the absence of novelty signal, these methods reduce to undirected exploration schemes. 
                
                