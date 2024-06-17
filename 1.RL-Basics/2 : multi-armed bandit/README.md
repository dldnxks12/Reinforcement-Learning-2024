### chapter 2 : MAB

---

- k-armed bandit
  
        # How to select best action?
          1. Estimate action values (Q)
          2. Select actions based on Q

         (1) action value estimation:
          1. Simple average ( Q(a) will converge to true Q(a) as n ➔ ∞ )
          2. Weighted average

         (2) action selection:
          1. greedy  : exploitation only
          2. ε-greedy: exploitation + exploration (random)
            ➔ all the Q values will converge to true Q as n ➔ ∞
          3. UCB
          4. gradient bandit
        
        # Incremental update        
        : value estimation 할 때, 모든 데이터들을 메모리에 저장해두었다가 계산하는 건 too expensive.
  
        ➔ 'incremental update' 
            ➔ Q(n+1) = Q(n) + (1/n)[R(n) - Q(n)]
            ➔ New-estimation = Old-estimation + Step-size[Target - Old-estimation]
            ➔ Q(n), (n), R(n)의 값만 저장해면 됨!


- Value estimateion : Sample Average vs. Weighted Average

        # Non-stationary problem ➔ dynamics change over time
          In Bandit : reward probabilities
          In MDP    : reward probabilities and transition kernel.
  
          so? let's give weight to the recent reward!
            ➔ Use a constant step size (α) ➔ i.e. Q(n+1) = Q(n) + (α)[R(n) - Q(n)], where α is constant.
            ➔ Weighted Average
    
        # Conditions of step size, required to converge ➔ Robbins-Monro Rule
          1. Sample Average (1/n): meets Robbins-Monro ➔ converge (Law of Large Number)
            ➔ Used in theoretical works
            ➔ Not useful in practice
          2. Weighted Average (α): violates Robbins-Monro ➔ not converge
            ➔ Useful for non-stationary problems.
        
        # Initial bias        
          1. Sample Average ➔ No initial bias  
          2. Weighted Average ➔ Induce initial bias
    
          Address the initial bias and non-stationary problems?
                ➔ Use 'unbiased constant step-size trick'!


- Upper Confidence Bound (UCB)

        # UCB?
        : Select actions according to their potential (Deterministically explore)
  
        # Limitation of UCB
          ➔ fails to work in non-stationary problems!
          ➔ fails to work in large state space!
          ➔ Bandit이 아닌 RL에 사용하기엔 ε-greedy가 더 낫다.

        # Improved version of UCB
          ➔ UCB 1  (if we dont have a prior knowledge about reward distribution)
          ➔ KL-UCB (if we have a prior knowledge about reward distribution)


- Gradient Bandit

        # Gradient Bandit can be seen as a stochastic approximation of gradient ascent
        # Choice of Baseline
                ➔ action에만 독립적이면 된다. 그 외에는 조건 X
  
                ➔ algorithm의 expected update of algorithm에는 전혀 영향을 미치지 않는다.
                ➔ 하지만, variance of the update에는 영향을 미치고, convergence rate에 영향을 미치게 된다.

---

- Sumamry : Value-estimateion Methods in Bandit

        1. Sample Average   (Use in stationary situation)
        2. Weighted Average (Use in non-stationary situtaion)
  
  
- Sumamry : Action Selection Methods in Bandit
        
        1. Greedy   : Select greedy one
        2. ε-greedy : Select greedy one, but sometimes get non-greedy actions 'with no preference' (Randomly explore)          
        3. UCB      : Select actions according to their potential (Deterministically explore)          
        4. Gradient Bandit : select actions by preference, not by reward

  
- Summary : Overall Problems 

        # K-armed Bandits (Non-associative) : 하나의 상황에서 action을 선택하는 문제
  
        # Contextual Bandits (Associative) : 상황에 맞게 action을 선택하는 문제
          ex. red-colored machine ➔ action 1
          ex. black-colored machine ➔ action 2
          - 선택된 action이 immediate reward에만 영향을 미침

        # Reinforcement Learning : 상황에 맞게 action을 선택하는 문제
          - 선택된 action이 미래의 reward와 state에 영향을 미침
                  
  
---

    
    


    

    







    


    
