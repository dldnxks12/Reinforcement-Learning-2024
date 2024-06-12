#### Reinforcement Learning (Sutton & Barto)


---

- Memorize 
     
   
       1) Deadly-Triad : brings instability & convergence issue.
       
           1) bootstrapping (e.g. DP / TD)
           2) off-policy    (e.g. Q-Learning) 
           3) non-linear FA (e.g. Neural Network)
   
   
       2) confusing MDP names
       
           1) Discrete MDP (finite MDP) vs Continuous MDP (infinite MDP)
           
               Discrete MDP   => the state, action, and reward sets are finite (discrete)
               Continuous MDP => the state, action, and reward are infinite (continuous)
           
           2) Deterministic MDP vs Stochastic MDP
               
               Deterministic MDP => deterministic policy
               Stochastic MDP    => stochastic policy

           
---


- DP vs MC vs TD
  

      1) DP, MC, TD 모두 control problem은 동일하고, prediction 방법만 다름. 
      
           DP : Perfect Model (o) + Bootstrapping (o)
           MC : Perfect Model (x) + Bootstrapping (x)
           TD : Perfect Model (x) + Bootstrapping (o)
   
      2) Optimal Policy

          * DP (Policy Iteration, Value Iteration)
            ➔ DP gurantees the optimal policy

          * MC (First visit, Every visit)
            ➔ MC gurantees the optimal policy

          * TD (Sarsa, Q learning)
            ➔ TD can not guarantee the optimal policy. (do converge but, not optimal)
           
       3) Approaches for reducing 'overestimation bias' in Q learning
   
           1. Double Q-learning or Clipped-double Q-learning
           2. Ensemble network
           3. Quantile regression
