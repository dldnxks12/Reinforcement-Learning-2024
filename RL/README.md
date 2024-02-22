#### Reinforcement Learning (Sutton & Barto)

---

    * Deadly-Triad : brings instability & convergence issue.
    
        1) bootstrapping (e.g. Q-learning / SARSA)
        2) off-policy    (target policy != behavior policy) 
        3) non-linear FA (e.g. neural network)


    * confusing MDP names
    
        1) Discrete MDP (finite MDP) vs Continuous MDP (infinite MDP)
        
            Discrete MDP   => the state, action, and reward sets are finite (discrete)
            Continuous MDP => the state, action, and reward are infinite (continuous)
        
        2) Deterministic MDP vs Stochastic MDP
            
            Deterministic MDP => deterministic policy
            Stochastic MDP    => stochastic policy 
        

    * DP vs MC vs TD
   
        DP : Perfect Model (o) + Bootstrapping (o)
        MC : Perfect Model (x) + Bootstrapping (x)
        TD : Perfect Model (x) + Bootstrapping (o)


    * Basic approaches of reducing 'overestimation bias'

        1. Double Q-learning or Clipped-double Q-learning
        2. Ensemble network
        3. Quantile regression
        
