### Chapter 4 : Dynamic Programming

---

- Dynamic Programming


      A collection of algorithms that can be used to compute optimal policies given a perfect model (MDP)

      # Limitation in utility
        1) A perfect model; p(s', r | s, a), is needed
        2) great computational expense

      # Typical assumption in DP
        1) Perfect Model is given.
        2) finite MDP

      >> if we find optimal v, q then we can recover optimal policy directly (i.e. greedy policy)



- Method

      # Policy Iteration (Based on Bellman Expectation Eqn.)
              Policy Evaluation >> Policy Improvement >> Policy Evaluation >> Policy Improvment >> ...

      # Value Iteration (Based on Bellman Optimality Eqn.)
              (Policy Evaluation + Policy Improvement) >> (Policy Evaluation + Policy Improvement) >> ...


      >> To overcome DP's problem; sweep on entire states for updating, we can turn into 'asynchronous DP' method.


      * Synchronouse DP (Classial DP)

            : Performing 'expected update' operation on each state.
              >> updates the value of one state based on the values of all possible successor states + probabilities of occurring.
  
      * Asynchronouse DP

            : In-place iterative methods that update states in an arbitrary order.



- DP's property : Bootsrapping


      It estimate value functions based on the basis of the other estimates.
  
      Therefore, DP is a method of 'bootstrapping'..!

        
