### chapter 5 : Monte Carlo Methods

---

- Monte Carlo methods properties

      * Assumptions in MC
        1) Perfect Model; p(s', r | s, a), is unknown

      * Limitation in MC 
        Only in episodic tasks.

---

- Monte Carlo Prediction (first-visit MC / every-visit MC)

      * DP vs MC
  
        Both of them follow the GPI framework (PE ➔ PI ➔ ...)
  
        However,
  
        DP evaluates v or q by 'computing' from known dynamics of MDP.
        MC evaluates v or q by 'learning' from sample returns in sample episodes.


      * Advantages of MC over DP.
        ➔ The estimation for each state is independent of other states (unlike DP).
          ➔ The estimation for one state does not build upon the estimation of any other state.
            ➔ That is, MC is not bootstrapping.
              ➔ The computational complexity of estimating v, q is independent to the number of states!


      * estimate v? or estimate q?

        Under the GPI framework, we need to improve the policy over approximated v/q.
        For this, it is better to estimate q 
            
            ➔ However, we have to ensure that all the state-action pairs have a non-zero visitation probability.
                ➔ Use 1. exploring starts + greedy policy  
                ➔ Use 2. soft-greedy policy


---

- Monte Carlo Control

      Policy improvement step 

      ➔ It requires assurance of visitation to find optimal policy when we work with q-function.
        ➔ Use 1. exploring starts + greedy policy  
        ➔ Use 2. soft-greedy policy


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

  

      
          



    

            
