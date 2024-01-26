#### delay

---

- Dynamics programming and optimal control

      *propose augmented approach 

---

- MDP with delays and asynchronous cost collection (2003)

      delayed MDP can be reduced to standard MDP using asynchronous cost 

---

- Planning and learning in environments with delayed feedback (2007)

      augmented approach has the issue of state-space explosion -> intractable
  
      *propose model-based simulation (MBS) 

---

- Control delay in RL for Real-Time dynamics systems: A memoryless approach (2010)

      memoryless method (state-dependent policy) : select action based only one the most recent observation 

      SARSA(λ) (memoryless method) performs good (when delay is small)
        -> Using eligibility traces to find the best memoryless policy in POMDPs (1998)
      Advantages    : 1) no need to augment state (augmented approach)
                      2) no estimation of explicit model (MBS)
  
      Disadvantages : 1) suboptimal (POMDP, resulting from delay)
                      2)doesn't take the delay into account in learning (state-dependent decision).

      *propose dSARSA(λ) (best performance among dSARSA, dSARSA(λ), dQ, and dQ(λ))
        -> improved SARSA(λ), which takes the delay into account.
        -> dSARSA, dSARSA(λ) have no guarantee on converge. But works well. 
        -> dQ do converge, but dQ(λ) := (dQ + eligibility traces) goes diverge.
          -> dQ is the only proved one to converge.

---
  
- At human speed : DRL with action delay (2018)      

      Using a state-predictive model as in MBS, but based on recurrent neural network.
       -> agent acts according to an estimate of the true state where their action will be executed. 

---
      
- Real-Time RL (2019)

    
  


      

  
  
      
