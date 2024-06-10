### chapter 5 : Monte Carlo Methods

---

- Quick Summary of MC 

      * Just as DP, MC 방법도 optiaml policy를 보장한다. *

      Assumptions in MC
      - Perfect Model; p(s', r | s, a), is unknown

      Limitation in MC 
      - only in episodic tasks.

      MC Methods is also in GPI framework
      - Prediction (policy evaluation)
            - First visit
            - Every visit
  
      - Control (policy improvement)
            - greedy + exploration starts
            - soft-greedy       
  

- Monte Carlo Prediction (first-visit MC / every-visit MC)

      # Policy evaluation step #

      * DP vs MC
  
        Both of them follow the GPI framework (PE ➔ PI ➔ ...)
  
        However,
  
        DP evaluates v or q by 'computing' from known dynamics of MDP.
        MC evaluates v or q by 'learning' from sample returns in sample episodes.


      * Advantages of MC over DP.
  
        The estimation for each state is independent of other states (unlike DP).
        ➔ The estimation for one state does not build upon the estimation of any other state.
        ➔ No bootstrapping
        ➔ The computational complexity of estimating v, q is independent to the number of states!


      * estimate v? or estimate q?

        Under the GPI framework, we need to improve the policy over approximated v or q.
        이를 위해서는 q를 추정하는게 더 낫다.
        왜? v를 추정하게 되면 policy improvement할 때, transition probability 정보가 필요.. 
            
        근데, q를 추정하면 문제가 하나 발생한다.
        즉, 모든 state-action pair들이 다 충분히 많이 sampling되어야한다는 것.
  
        SO?
        1) Use greedy policy + exploring starts
        2) Use soft-greedy policy


- Monte Carlo Control

      # Policy improvement step #

      Policy improvement를 위해서는 given policy에 대한 정확한 q 값이 필요.
      그리고 다시 말하지만, 정확한 q를 위해서는 아래의 두 방법을 사용한다.

      1) Use greedy policy + exploring starts
      2) Use soft-greedy policy
    
      이를 통해 정확하게 추정된 q가 있어야 optimal policy를 찾아낼 수 있다.


- Monte Carlo Control : Off-policy prediction


      * On-policy vs Off-policy

          On-policy  : Fast convergence and simple
          Off-policy : Slow convergence and high-variance ➔ But powerful! 
          (e.g., learn from data of expert policy)
    
      * 'Assumption of coverage' in off-policy learning

          In order to use episodes from behavior policy to estiamte values under the target policy,
          we required that every action taken under target policy is also taken under behavior policy. 


      How can we learn about the optimal policy, while behaving according to an exploratory policy?

      Actually, 'on-policy' learning is a compromise..
      ➔ It learns q-functions not for the optimal policy, but for a near-optimal policy that still explores!!

      Instead, we can use two policies for learning optimal policy while exploring
        ➔ one is for learning (target policy), and the other is for exploring (behavior policy)!
        ➔ This is an 'off-policy' learning.

  

      
          



    

            
