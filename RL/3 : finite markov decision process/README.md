### chapter 3 : finite MDP

---

        Finite Markov Decision Process (Sequential decision making problem)
        ➔ actions influence not just immediate rewards, but also sebsequent states.
        ➔ finite MDP : the sets of states, actions, rewards have a finite number of elements.
        ➔ associative : choose different actions at different situations. 
        (non-associative : chooese optimal action in a same situation (bandits)) 

        Bandits ➔ estimate the value Q(a) of each action (a)
        MDPs    ➔ estimate the value Q(s, a) of eacth action (a) at the state (s)

---


- Agent-Env. interface
    

        # dynamics of MDP (p)
            p(s', r | s, a) (이게 MDP의 dynamics라는 걸 이제 알았다.)

        with dynamics of MDP (p), we can figure out anything about env. 
            ➔ p(s' | s, a) : state transition probabilities
            ➔ r(s, a, s')  : the expected rewards for state-action-next_state triples
            ➔ r(s, a)      : the expected rewards for state-action pairs

        >> MDP 환경을 안다? 1) transition probability 2) reward structure 를 안다는 것.



            
