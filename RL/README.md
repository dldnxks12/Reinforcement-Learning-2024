#### Reinforcement Learning (Sutton & Barto)

---

- Part 1 flow :

      [1] Introducing the Bandit environment and Finite MDP environment 
      [2] Three fundamental classes of methods for solving finite MDP
         >> Dynamic Programming 
         >> Monte Carlo 
         >> Temporal Difference

      [3] Combining DP, MC, and TD

---

- Memorize : 

     
   
       * Deadly-Triad : brings instability & convergence issue.
       
           1) bootstrapping (e.g. DP / TD)
           2) off-policy    (e.g. Q-Learning) 
           3) non-linear FA (e.g. Neural Network)
   
   
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
   
   
       * Basic approaches for reducing 'overestimation bias'
   
           1. Double Q-learning or Clipped-double Q-learning
           2. Ensemble network
           3. Quantile regression

      * Optimal Policy

            * DP (Policy Iteration / Value Iteration)
            >> We can always find the optimal policy in MDP with a greedy improvement policy.

            * MC
            >> We can find the optimal policy in MDP (verify further in the future).

            * TD
            >> JW said, TD can not guarantee to find the optimal policy in MDP.
  
           
