### chapter 5 : Monte Carlo Methods

---

- Monte Carlo methods properties


      1) No complete knowledge of MDP.
      2) Only in episodic tasks.


- Monte Carlo Prediction (first-visit MC / every-visit MC)

      * DP vs MC in v, q estimation.
  
        Both of them follow the GPI framework
  
        However,
  
        DP evaluates v or q by 'computing' from known dynamics of MDP.
        MC evaluates v or q by 'learning' from sample returns in sample episodes.


      * Advantages of MC over DP.
        >> The estimation for each state is independent of other states (unlike DP).
        == The estimation for one state does not build upon the estimation of any other state.
          >> That is, MC is not bootstrapping.
            >> The computational expense of estimating v, q is independent of the number of states!
  


      * estimating v? q?

        Under the GPI framework, we need to improve the policy over approximated v/q.
        For this, it is better to estimate q (action-value function)

          >> However, we have to ensure that all the state-action pairs have a non-zero visitation probability.
            >> Use 1) exploring starts 2) using soft-greedy policy.


- Monte Carlo Control


      Policy improvement based on estimated q-function.

      * Required assurance of visitation for finding optimal policy based on q-function.
  
            method 1) exploring starts + greedy policy
            method 2) soft-greedy policy


- Monte Carlo Control : Off-policy prediction


      How can we learn about the optimal policy, while behaving according to an exploratory policy?

      Actually, 'on-policy' learning is a compromise..
      >> It learns q-functions not for the optimal policy, but for a near-optimal policy that still explores!!

      Instead, we can use two policies for learning optimal policy while exploring
      >> one is for learning (target policy), and the other is for exploring (behavior policy)!
        >> This is an 'off-policy' learning.

  
      * On-policy vs Off-policy

          On-policy  : Fast convergence and Simple
          Off-policy : Slow convergence and High-variance >> But powerful! 
          (e.g. learn from data of expert policy)
    
      * 'Assumption of coverage' in off-policy learning

          In order to use episodes from behavior policy to estiamte values under the target policy... 
          we required that every action taken under target policy is also taken under behavior policy. 

      
          



    

            
