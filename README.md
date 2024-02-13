#### Histories (constant / random)

---

- Closed-loop control with delayed information (ACM 1992)

      *propose an 'augmented approach'
       -> but, it can not scale up to stochastic delay (i.e. random delay)

---

- MDP with delays and asynchronous cost collection (IEEE TAC 2003)

      Delayed MDP can be reduced to standard MDP using asynchronous cost.
      In addition, it scales up to stochastic delay with 3 considerations
       1. infinite state length -> freeze
       2. reward duplication -> collect cost, once.
       3. reward discount -> use r0
  
       -> but, freezing of execution has a limitation of real-time systems.

---

- Planning and learning in environments with delayed feedback (2007)

      augmented approach has the issue of state-space explosion -> intractable
  
      *propose 'model-based simulation (MBS)'

       -> but, it is hard to apply to stochastic MDPs.

---

- Control delay in RL for Real-Time dynamics systems: A memoryless approach (IEEE/RSJ 2010)

      memoryless method (state-dependent policy) : select an action based only on the most recent observation 

      SARSA(λ) (memoryless method) performs well (when the delay is small)
        -> Using eligibility traces to find the best memoryless policy in POMDPs (1998)
      Advantages    : 1) no need to augment state (augmented approach)
                      2) no estimation of explicit model (MBS)
  
      Disadvantages : 1) suboptimal (POMDP, resulting from delay)
                      2) doesn't consider the delay in learning (state-dependent decision).

      *propose 'dSARSA(λ)' (best performance among dSARSA, dSARSA(λ), dQ, and dQ(λ))
        -> improved SARSA(λ), which takes the delay into account.
        -> dSARSA, dSARSA(λ) have no guarantee on converge. But works well. 
        -> dQ do converge, but dQ(λ) := (dQ + eligibility traces) goes diverge.
          -> dQ is the only proved one to converge.

       -> but, it can not scale up to stochastic delay.

---
  
- At human speed : DRL with action delay (2018)      

      Using a state-predictive model as in MBS, but based on a recurrent neural network.
       -> Agent acts according to an estimate of the true state where their action will be executed. 

---
      
- Real-Time RL (NIPS 2019)

      Most RL algorithms assume that the state of the agent's environment (MDP) does not change during action selection.
      (policy = agent | environment = MDP)

      *propose a new framework 'RTRL' where the state and action change simultaneously and propose 'RTAC'

---

- RL with random delays (ICLR 2021)

---

- Blind decision making: RL with delayed observations (ICAPS 2021)


      *propose Expectated-Q Maximization (EQM)

       EQM is
       1) Space efficient -> better than basic augmented approach 
       2) Robust under deviation from most likely state -> better than MBS
       3) Handles constant and stochastic delays, both 


---

- DIDA

---

- BPQL

---

- AD-RL

---





    
  


      

  
  
      
