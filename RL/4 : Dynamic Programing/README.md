### Chapter 4 : Dynamic Programming

---

- Dynamic Programming

      *******************************************************************************

      We find value function / optimal value function by solving Bellman Equations.

      v(π) ➔ Bellman Equation ➔ solve 'system of linear equation'
      v(*) ➔ Bellman Optimality Equation ➔ solve 'system of non-linear equation'

        * optimal value function 찾으면 optimal policy는 easy하게 찾아진다. (e.g., greedy)

      MDP dynamics; p가 주어져있다면, 그냥 이렇게 푸는게 straightforward.
      근데, computational burden + instability 문제가 있으니 iterative 하게 풀어보자 ➔ 'DP'

        * Instability issue : Matrix Inversion
      
      *******************************************************************************


        * Assumptions in DP
          1) Perfect Model; p(s', r | s, a), is given.
          2) finite MDP 

        * Limitation
          1) A perfect model; p(s', r | s, a), is needed
          2) Intractable in inifinite MDP

---

- GPI

      [1] Policy Iteration (Based on Bellman Eqn.) 
      
        = Policy Evaluation ⮂ Policy Improvement

          ➔ We can get monotonically improving π and v.
            ➔ this guarantees to converge to the optimal π and optimal v.
  
        * Policy Evaluation : v
              ➔ Iterative policy evaluation
                ➔ v(π) exist and is unique (c.f., Ok's lecture)
                ➔ v(k) converges to v(π), as k ➔ ∞

        * Policy Improvement : π
              ➔ greedy policy (meets policy improvement theorem)
              ➔ soft-greedy policy (meets policy improvement theorem)

      
      [2] Value Iteration (Based on Bellman Optimality Eqn.)

        = (Policy Evaluation + Policy Improvement) ⮂ (Policy Evaluation + Policy Improvement)


      *************************************************************************

      RL은 GPI framework를 따른다. 
      DP를 이용한 GPI는 convergence가 증명이 되어있다. 하지만, 다른 방법을 이용한 GPI는 아직...

      *************************************************************************



---

- Convergence of Policy Iteration and Value Iteration



      *************************************************************************

      Bellman Operator B is a 'Contraction Mapping'. 
          ➔ Contraction Mapping implies..
            1) Convergence 
            2) Uniqueness

      But, the converged point is the optimal point?
      No! we need greedy improvement policy such as 'greedy policy' 
          ➔ 'Monotonic Improvement Condition'!

      * PI and VI with 'greedy improvement' always find optimal policy!
          ➔ it is thanks to 'Markov Property' in MDP 

      *************************************************************************


---

- Drawbacks

      The major drawback of DP is... they need to sweep entire states for updating.
        ➔ to overcome this, we can employ the 'asynchronous DP' method

      * Synchronous DP (Classical DP)

        ➔ Performing 'expected update' operation on each state.

      * Asynchronous DP

        ➔ In-place iterative methods that update states in an arbitrary order.
        ➔ 결국에 converge하려면 state를 다 sweep 해야하지만, 무엇을 먼저 update할 지 결정할 수 있어 flexible. 


---

- DP's property: Bootstrapping

      It estimates value functions based on the other estimates.
  
      Therefore, DP is a method of 'bootstrapping'..!

        
