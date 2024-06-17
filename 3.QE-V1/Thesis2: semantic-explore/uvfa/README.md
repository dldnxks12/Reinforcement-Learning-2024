#### Universal Value Function Approximator (UVFA)

---


- Brief summary of UVFA


        * 1 line summary

            UVFA is a value function approximator that generalize over the state 's' and goals 'g' together.


        * Key idea  

            The main idea of UVFA is to represent a large set of optimal value functions 
            by a single, unified function approximator that generalizes over 'both states and goals'.

                V(s,g; θ) : universal value function approximator 
                V_g(s) : general value function

                ➔ V(s,g; θ) ≈ V_g(s), where V_g(s) is a ground-truth value function of V(s,g; θ)

        * Method : two-stage regression based on factorization
            
            make a sparse table of values that contains rows for each observed state s and columns for each observed goal g.
                ➔ find a low-rank factorization of the table into state embeddings φ(s) and goal embeddings ψ(g)
                    ➔ learn non-linear mappings from state s to φ(s), using MLP 
                    ➔ learn non-linear mappings from goal g to ψ(s), using MLP

            using the learned mapping functdions (MLPs), we can infer V(s,g; θ) even for unseen states and goals!
        

        * Experiments : 1) supervised-learning setting and 2) reinforcement learning setting.

            