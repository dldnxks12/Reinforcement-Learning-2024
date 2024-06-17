#### Never Give Up (NGU)

---

- Brief summary of NGU


            * 1 line summary

                they propose the NGU intrinsic reward, that does not 'vanish' over time!

            * A core contribution 

                a method for learning policies that can maintain exploration throughout the training process. 
                ➔ In Depth-First-Search (DFS) mannar

            * key factors
    
                1) novelty signal이 사라지지 않는 intrinsic reward 제안 (per-episode novelty + life-long novelty)
                2) self-supervised inverse dynamic model을 이용한 controllable state 추출 (like ICM)
                3) UVFA를 이용한 exploration & exploitation policy family들 학습 (β_0 = 0, ..., β_N = β)
                4) intrinsic reward로 인해 MDP ➔ POMDP ➔➔ history information 이용하는 RL framework 차용 (R2D2)


--- 


            * Exploration problems in RL 

            Methods that guarantee finding an optimal policy require a sufficient number of visitations to each state-action pair.
            ➔ ensuring that all state-action pairs are encountered infinitely often is the general problem of maintaining exploration.

            The simplest approach : stochastic policies with a non-zero prob. of selecting all actions in each state.
            ➔ ε-greedy or Boltzmann exploration
            ➔ these methods will eventually learn the optimal policy in 'tabular-setting'.
                ➔ but, very inefficient and the steps they require grow exponentially with the size of the state space.

            Despite these disadvantages, they perform well in 'dense' reward settings.
            ➔ but, in 'sparse' reward setting, they fail to learn
                ➔ because it is hard to find the very few rewarding/meaningful states.
                (temporally-extended exploration or deep exploration is crucial)    


            ➔ ➔ ➔ ➔ ➔ Intrinsic reward is proposed to overcome exploration issues in 'sparse' reward & 'non-tabular' settings,


---

            * Intrinsic reward setting (1)

            : 현재의 state가 지금가지 방문한 state와 얼마나 다른 지를 측정해서, 이 차이를 intrinsic reward로 이용.

                * 한계 (1)
                : state의 novelty가 사라지면 intrinsic reward를 이용한 exploration 중단됨 ➔ undirected exploration


            * Intrinsic reward setting (2)
        
            : World Model이나 Inverse Dynamic Model 만들고, prediction error를 이용해서 intrinsic reward 설계 (c.f ICM)
                
                * 한계 (2)
                : explicitly building predictive models, particulary from observations, is expensive and error prone.
                그리고 얘도, 모델 학습이 정확해질수록 novelty signal이 소멸 ➔ undirected exploration


            * 정리

            In the absence of novelty signal, these methods reduce to undirected exploration schemes
            ➔ turn to undirected exploration  
                
                