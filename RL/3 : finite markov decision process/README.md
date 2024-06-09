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
        


- Key notes of Bellman Equations (quick summary)

        1. v(s) = E[Gt|st]를 recursive하게 표현한 식?
                = E[Rt+1 + γv(St+1)|st] = Bellman Equation !
                  
        2-1. v(s)는 Bellman Equation의 유일한 해 (Unique solution)
        2-2. v*(s)는 Bellman Optimal Equation의 유일한 최적해 (Unique solution & Optimal solution)  

        3. 우리는 Bellman Equation들을 걍 다이렉트하게 풀어버릴 수도 있다.
  
        ➔ Bellman Equation은 system of linear equation이므로 Matrix Inversion 이용해서.
        하지만 this is not 만만...
  
        SO, 아래의 세 가지 방법을 이용해서 v(s)를 찾아내는 방법을 자주 사용한다.
        ➔ Dynamic Programming 
        ➔ Monte Carlo
        ➔ Temporal Difference 

        '위 세 가지 방법을 Bellman Equation의 stochastic approximation이라고 한다.'


- Bellman Equations (Expection / Optimal)
        
        (1) Bellman Expectation Equation
  
        v(s) and q(s,a) can be expressed in recursive form.
        이 recursive form이 바로 'Bellman Equation'이다!
        그리고, Bellman Equation은 'system of linear equation'이다.
 
        We can find v(s) and q(s,a) by solving this 'system of linear equations'.
        ➔ by Matrix Inversion !
        ➔ but this approach is quite not favorable due to the instability.


        (2) Bellman Optimality Equation

        The v*(s) and q*(s) can also be expressed in recursive form.
        이 식은 'Bellman Optimality Equation'!  
        Bellman Optimality Equation도 역시 'system of non-linear equation'이다.
      
        We can find v*(s) and q*(s,a) by solving this 'system of non-linear equations'.
  
        If we get v* and q*, then we can find optimal policy straight-forwardly!
        왜? 최적의 value들을 찾았으니, 이를 바탕으로 행동을 결정하면 됨. 당연한 것.
        SO?
        ➔ v* 를 찾았을 때, do one-step search for finding optimal policy. (select greedy action)
        ➔ q* 를 찾았을 때, we don't need to do one-step search!    

        
        ***************************************************************************************************

        * Bellman Optimality Equation (BOE)을 풀면 Optimal Policy를 찾을 수 있다.
          (BOE를 푼다 == BOE를 만족시키는 v*, q*를 구한다.)
  
          즉, 강화학습 문제가 풀리는 것이다. 

          하지만, 당연히 쉽지 않겠지? BOE를 explicit하게 풀려면 다음과 같은 세 가지 가정을 충족해야한다.
            1) MDP dynamics; p 를 알아야한다. 
            2) Sufficient computational resources 
            3) the states have 'Markov Property' (Markovian state)

        일반적으로, 3)은 만족이 되는데, 1), 2)가 만족이 안된다. 
        그래서, 우리는 BOE를 explicit하게 푸는 것 대신 BEO의 stochastsic approximation을 통해 solution을 찾아낸다.
        (ex. Dynamic Programming, Monte Carlo, Temporal Difference)

        ***************************************************************************************************

---








        








            
