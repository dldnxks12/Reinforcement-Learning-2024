### chapter 5 : Monte Carlo Methods

---

- Monte Carlo methods properties


      1) No complete knowledge of MDP.
      2) Only in episodic tasks.


- Monte Calro Prediction (first-visit MC / every-visit MC)

      * DP vs MC in v, q estimation.
  
        Both of them follows GPI framework
  
        However,
  
        DP evaluates v or q by 'computing' from known dynamics of MDP.
        MC evaluates v or q by 'learning' from sample returns in sample episodes.


      * Advantages of MC over DP.
        >> The estimation for each state are independent on other states (unlike DP).
        == The estimation for one state does not build upon the estimation of any other state.
          >> That is, MC is not bootstrapping.
            >> The computational expense of estimating v, q is independent of the number of states!
  


      * estimating v? q?

        Under the GPI framework, we need to improve the policy with respect to approximated v/q.
        For this, it is better to estimate q (action-value function)

          >> However, we have to ensure that all the state-action pairs have a non-zero visitation probability.
            >> Use 1) starting exploration method 2) using soft-greedy policy.


- Monte Carlo Control 
