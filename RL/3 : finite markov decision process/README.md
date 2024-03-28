### chapter 3 : finite MDP

        Markov Decision Process
        : Mathematically idealized form of the RL problem.
            ➔ We can make precise theoretical statements in this framework! 

---

- Finite MDP

        Finite Markov Decision Process (Sequential decision-making problem)
        ➔ actions influence not just immediate rewards, but also subsequent states.
        ➔ finite MDP: the sets of states, actions, and rewards have finite elements.
        ➔ associative: choose different actions in different situations. 
                ➔➔ non-associative: choose optimal action in the same situation (bandits)

            Bandit : estimate the value Q(a) of each action (a)
            MDP    : estimate the value Q(s, a) of each action (a) at the state (s)



---


- Agent-Env. Interface
    

        Dynamics of MDP (p)

            p(s', r | s, a) (이게 MDP의 dynamics라는 걸 이제 알았다..)

            with dynamics of MDP (p), we can figure out "anything" about env...!!

            ➔ p(s' | s, a) : state transition probabilities
            ➔ r(s, a, s')  : the expected rewards for (state - action - next state) triples
            ➔ r(s, a)      : the expected rewards for (state - action) pairs

        MDP 환경? 
        ➔➔ transition probability 
        ➔➔ reward structure


---

- Bellman Equation

        The v(π) and q(π) can be expressed in recursive form (or Self-Consistency).
        This form is 'Bellman Equation'!
          ➔ Bellman Equation is a system of equation (linear).
            (n states unknown ➔ n equations unknown) 
 
        If we know the MDP dynamics; p, 
        we can find v(π) and q(π) by solving this 'system of linear equations'.
          ➔ ex1. Matrix Inversion (this approach is quite not favorable due to the instability) 
          ➔ ex2. Iterative method (Dynamic Programming)


- Bellman Optimality Equation

        The v*(π) and q*(π) can also be expressed in recursive form (or Self-Consistency).
        This form is 'Bellman Optimality Equation'!
          ➔ Bellman Optimality Equation is a system of equation (non-linear).
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
            3) the states have 'Markov Property'

        따라서, 우리는 BOE를 explicit하게 푸는 것 대신 approximate solution을 찾아낸다. (ex. Dynamic Programming)
        (Most of the methods can be viewed as ways of approximately solving this Bellman Optimality Equation.)
        ***************************************************************************************************************

---


- Optimality and Approximation







        








            
