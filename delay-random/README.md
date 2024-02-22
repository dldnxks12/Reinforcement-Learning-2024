#### Summary of random delay 

---

- `Tentative idea`

      1. Train multiple BPQL Agent corresponding to each constant delay
        >> BPQL agent for 2 time-step delay, for 3 time-step delay, etc

      2. GPR / BO based delayed time-step agnostic
        >> Estimate delayed time-steps

      3. Select BPQL Agent corresponding to estimated delayed time-steps.
        >> estimated delayed time-step is 3? select BPQL Agent correspoding to it!

--- 

- Augmented Approach in Random delay 

      # MDP with delays and asynchronous cost collection (IEEE TAC 2003)

      Transform MDP with delays into MDP without delays by augmenting state.
      This paper found that we don't need to use this complex cost-structure!!
      >> The cost can be collected asynchronously as long as they are discounted properly.
      

---

- DC/AC

      # Reinforcement learning with random delays (ICLR 2021)

---

- DIDA

---

- BPQL (constant)

---

- AD-RL

---



