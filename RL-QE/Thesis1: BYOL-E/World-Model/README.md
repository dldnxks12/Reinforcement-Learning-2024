#### World-Model

----

- Brief sumamry


      # 1 line sumamry 
      It explore the 'generative model' in RL.

      # The model consists of 3 components
        ➜ Visual Sensory Component (V model)  : Compress each observed input frames (images) into a small representation vector z.
        ➜ Memory Component (M model)          : Make predictions about future feature vectors based on historical information
          ➜➜ Also acts like a predictive model of the future z
          ➜➜ we can play by generating hallucination!
        ➜ Decision-Making Component (C model) : Decide what actions to take based on the representations from V and C models.

      # world model = (V model + M model)

  

  
  

    

      

      



