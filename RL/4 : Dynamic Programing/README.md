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

      >> If we find optimal v, q then we can recover optimal policy directly (i.e. greedy policy)



- Method

      # Policy Iteration (Based on Bellman Expectation Eqn.)
              Policy Evaluation >> Policy Improvement >> Policy Evaluation >> Policy Improvement>> ...

                    * Policy Evaluation
                          1) Using Matrix Inversion (Inverse always exist-Perron-Probenious, but this is a highly unstable method)
                          2) Dynamic Programming: iterative method (guaranteed by contraction mapping)

                    * Policy Improvement
                          1) greedy policy
                          2) soft-greedy policy

      # Value Iteration (Based on Bellman Optimality Eqn.)
              (Policy Evaluation + Policy Improvement) >> (Policy Evaluation + Policy Improvement) >> ...


      >> To overcome DP's problem (sweep entire states for updating) >> we can turn to the 'asynchronous DP' method.


      * Synchronous DP (Classical DP)

            : Performing 'expected update' operation on each state.
              >> updates the value of one state based on the values of all possible successor states + probabilities of occurring.
  
      * Asynchronous DP

            : In-place iterative methods that update states in an arbitrary order.



- DP's property: Bootstrapping


      It estimates value functions based on the other estimates.
  
      Therefore, DP is a method of 'bootstrapping'..!

        
