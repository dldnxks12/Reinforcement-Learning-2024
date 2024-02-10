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
