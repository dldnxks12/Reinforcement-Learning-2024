#### Summary of constant delay 

---

- Augmented Approach

      # MDP with delays and asynchronous cost collection (IEEE TAC 2003)

      DDMDP can be reduced to standard MDP using asynchronous cost collection.
      SDMDP can also be reduced to standard MDP using asynchronous cost collection.
      >> It proposes an approach for the random delay (accompanied by 3 considerations)
         1. state explosion     >> freeze
         2. cost duplication    >> collect cost, just once.
         3. cost discount       >> use r0

      >> but, this framework assumes the strong assumption of 'ordering'
              >> the state s(t+1) can not be processed before s(t) 
      >> Also, freezing of execution has a limitation of real-time systems.

---

- MBS 

      # Planning and learning in environments with delayed feedback (2007)

      The augmented approach has the issue of 'state-space explosion' 
  
      * Proposed 'model-based simulation (MBS)'
      >> but, it is hard to apply to stochastic MDPs (stochastic policy).

---

- dSARSA(λ)

      # Control delay in RL for Real-Time dynamics systems: A memoryless approach (IEEE/RSJ 2010)

      *memoryless method? select an action based only on the most recent observation (state-dependent policy). 

      SARSA(λ) performs well (only when the delay is small)
      >> Eligibility traces were used to find the best memoryless policy in POMDPs (1998)

      Advantages    : 1) no need to augment state (augmented approach)
                      2) no estimation of explicit model (MBS)
  
      Disadvantages : 1) suboptimal (POMDP, resulting from delay)
                      2) doesn't consider the delay in learning (state-dependent decision).

      *Proposed 'dSARSA(λ)' (best performance among dSARSA, dSARSA(λ), dQ, and dQ(λ))
        >> improved SARSA(λ) by taking the delay into account.
        >> dSARSA, dSARSA(λ) have no guarantee on converge. But works well...! 
        >> dQ guarantees converge, but dQ(λ) := (dQ + eligibility traces) goes diverge.
          >> dQ is the only proved one to converge.

       >> but, dSARSA(λ) can not scale up to random delay.

---
  
- At human speed : DRL with action delay (2018)      

      Using a state-predictive model as MBS, but based on a recurrent neural network.
      >> Agent acts according to an estimate of the true state where their action will be executed. 

---

- EQM

      # Blind decision making: RL with delayed observations (ICAPS 2021)
      
      * Proposed Expectated-Q Maximization (EQM)

       EQM is
       1) Space efficient -> better than basic augmented approach 
       2) Robust under deviation from most likely state -> better than MBS
       3) Handles constant and stochastic delays, both 

---

- RTAC (constant)

      # Real-Time RL (NIPS 2019)

      * Proposed RTAC in 'RTRL' framework (state and action changes simultaneously).

---
    
  


      

  
  
      
