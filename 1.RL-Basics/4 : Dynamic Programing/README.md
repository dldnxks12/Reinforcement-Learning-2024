### Chapter 4 : Dynamic Programming

---

- Recap. value function and Bellman Eqns.

  
      We can find v(s) and v*(s) by solving BEE and BOE.

      v(s)  ➔ BEE ➔ directly solve 'system of linear equation'
      v*(s) ➔ BOE ➔ directly solve 'system of non-linear equation'

      근데, Bellman equation을 direct하게 풀어내는 것은 일반적으로 많은 어려움이 있다.  
      SO, v(s) and v*(s)를 Bellman equation의 stochastic approximation으로 구하자.
  
      ➔ 'Dynamic Programming'
        
      * Assumptions in DP
      1) Perfect Model; p(s', r | s, a), is given.
      2) finite MDP 
     
      * Limitation
      1) A perfect model; p(s', r | s, a), is needed
      2) Intractable in inifinite MDP


- Dynamic Programming


      Dynamic Programming is 'iteration based stochastic approximation method'.

      (1) Policy Iteration (Based on BEE) 
      
        = Policy evaluation ➔ Policy improvement 반복
      
      (2) Value Iteration (Based on BOE)

        = (Policy evaluation + Policy improvement) 반복


      * Policy Evaluation : v(k) ➔ v(k+1)
        ➔ v(k) converges to v(π), as k ➔ ∞
        ➔ v(π) exist and is unique (c.f., Ok's lecture)
          ➔ Bellman Operator 'B' is a contraction mapping!
     
      * Policy Improvement : π(k) ➔ π(k+1)
        ➔ greedy policy 
        ➔ soft-greedy policy 
          ➔ policies that meet policy improvement theorem (monotonic improvement)


- Convergence of Dynamic Programming 


      *************************************************************************
      * Convergence to optimal policy
  
      1) contraction mapping이다.
        ➔ unique solution 보장.
      2) monotonic improvment를 보장한다.
        ➔ optimal solution을 보장.

      즉, 위 성질을 가진 iteration들을 계속하면 optimal π 그리고 optimal v로 수렴 !
      
      * contraction mapping? monotonic improvment?

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

      RL은 GPI framework를 따른다. 
      DP를 이용한 GPI는 optimal로의 convergence가 증명이 되어있다. 
      하지만, 다른 방법을 이용한 GPI는 아직..

      *************************************************************************


- Limitations of Dynamic Programming

      1) Need to know dynamics (model); p
      2) Great computational burdens
 
      The major drawback of DP is... they need to sweep entire states for updating.
        ➔ to overcome this, we can employ the 'asynchronous DP' method

      * Synchronous DP (Classical DP)

        ➔ Performing 'expected update' operation on each state.

      * Asynchronous DP

        ➔ In-place iterative methods that update states in an arbitrary order.
        ➔ 결국에 converge하려면 state를 다 sweep 해야하지만, 무엇을 먼저 update할 지 결정할 수 있어 flexible. 


---

