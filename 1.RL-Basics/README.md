#### Reinforcement Learning (Sutton & Barto)


---

- Memorize 
     
   
       1) Deadly-Triad : brings instability & convergence issue.
       
           1) bootstrapping (e.g. DP / TD)
           2) off-policy    (e.g. Q-Learning) 
           3) non-linear FA (e.g. Neural Network)
   
   
       2) confusing MDP names
       
           1) Discrete MDP (finite MDP) vs Continuous MDP (infinite MDP)
           
               Discrete MDP   ➔ the state, action, and reward sets are finite (discrete)
               Continuous MDP ➔ the state, action, and reward are infinite (continuous)
           
           2) Deterministic MDP vs Stochastic MDP
               
               Deterministic MDP ➔ deterministic policy
               Stochastic MDP    ➔ stochastic policy

           3) Non-stationary MDP ➔ dynamics change over time               
  
               In Bandit : reward probabilities
               In MDP    : reward probabilities and transition kernel.

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

---

- Deep RL


    * Value function parameterization

        DQN ➔ DDQN ➔ D3QN 

    * Policy function parameterization
    
        PG ➔ DPG ➔ DDPG ➔ TD3
           ➔ SQL ➔ SAC 
           ➔ TRPO ➔ PPO 


---

- Memorize 2 (c.f. SAC paper)


        # Limitations of Model-free Deep RL

            - Sample complexity
            - Sensitivity to Hyperparmeter

        # on-policy vs. off-policy (model-free deep RL)

            - on-policy  : stable   but sample inefficient
                ➔ sample이 충분하다면 'PPO'가 최고의 알고리즘일 것.
    
            - off-policy : instable but sample efficient
                ➔ sample이 부족하다면, 'SAC'를 쓰자.
                (TD3도 좋은데, SAC보다 hyperparameter에 민감하다.)

        # Off-policy Actor-Critic

            - Deterministic Actor ➔ TD3
            - Stochastic Actor    ➔ SAC

        # Actor-Critic architecture

            Value network와 Policy network를 독립적으로 구성하는게 더 안정적이다. 

        # Soft MDP 장점

            Maximum Entropy RL ➔ improve exploration and stability 

    





        

        

