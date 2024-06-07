### chapter 3 : finite MDP
---

- Relationship between RL and MDP
  
        MDP는 Agent-Environment로 표현되는 RL 환경을 모델링하기 위한 수학적인 framework.
        그리고, RL은 이 framework 안에서 최적의 정책을 찾는 알고리즘 또는 테크닉을 의미
  
        # Reinforcement Learning  = Markov decision process를 푸는 방법
                                  ➔ Learn sequential decision making process for given 'Markovian states'.
  
        # Markov decision process = Mathematically idealized form of the RL problem.                                
                                  ➔ We can make precise theoretical statements in this framework! 
                                  ➔ MDP를 governing하는 dynamics가 Markovian property를 가정하고 있는 것이 특징. 
                  

- Dynamics of MDP (p)

        p(s', r | s, a) (이게 MDP의 dynamics라는 걸 이제 알았다..)
        with dynamics of MDP (p), we can figure out "anything" about env...!!
        
        ➔ p(s' | s, a) : state transition probabilities
        ➔ r(s, a)      : the expected rewards for (state - action) pairs
        

- Bellman Equations (Expection / Optimal)
        
        Markovian property에 이은, RL의 또하나의 fundamental한 property?
        Value function과 Q function이 Bellman Equation들을 만족한다는 것.

        # Bellman Expectation Equation
  
        v(π) and q(π) can be expressed in recursive form ! (Self-Consistency)
        이 recursive form이 바로 'Bellman Equation'이다!
        ➔ Bellman Equation is a 'system of linear equation'.
 
        If we know the MDP dynamics; p, 
        we can find v(π) and q(π) by solving this 'system of linear equations'.
          ➔ ex1. Matrix Inversion (this approach is quite not favorable due to the instability) 
          ➔ ex2. Iterative method (Dynamic Programming)


        # Bellman Optimality Equation

        The v*(π) and q*(π) can also be expressed in recursive form ! (Self-Consistency)
        This form is 'Bellman Optimality Equation'!
        ➔ Bellman Optimality Equation is a 'system of non-linear equation'.
            (n states unknown ➔ n equations unknown)
      
        If we know the MDP dynamics; p,
        we can find v* and q* by solving this 'system of non-linear equations'.

        If we get v* and q*, then we can find optimal policy straight-forwardly!
          ➔ v* is known ➔➔ do one-step search for finding optimal policy.
          ➔ q* is known ➔➔ we don't need to do one-step search!
            ➔ We can find out the optimal policy without knowing the MDP dynamics p!

        
        ***************************************************************************************************************
        * Bellman Optimality Equation (BOE)을 풀면 Optimal Policy를 찾을 수 있다. *
          즉, 강화학습 문제가 풀리는 것이다. 

              * BOE를 푼다 == BOE를 만족시키는 v*, q*를 구한다.

          하지만, 당연히 쉽지 않겠지? BOE를 explicit하게 풀려면 다음과 같은 세 가지 가정을 충족해야한다.
            1) MDP dynamics is given 
            2) Sufficient computational resources 
            3) the states have 'Markov Property' (Markovian state)

        따라서, 우리는 BOE를 explicit하게 푸는 것 대신 approximate solution을 찾아낸다. (ex. Dynamic Programming)
        (Most of the methods can be viewed as ways of approximately solving this Bellman Optimality Equation.)
        ***************************************************************************************************************

---


- Optimality and Approximation







        








            
