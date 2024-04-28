#### BYOL

- Bootstrap Your Own Latent (BYOL)

----

- Brief Summary

    
        * 1 line summary
            ➔ BYOL is a self-supervised representation learning (image) 
            that learns its representation by predicting 'previous versions' of its outputs.  
            

        * concept
            ➔ define two networks : online network and target network.
                ➔ target network is a soft-copy of online network (moving-averaged-weight propagation)
            ➔ online network is trained to predict the outputs of target network 

            * online network outputs : representation of augmented image 
            * target network outputs : representation of augmented image from anther view of same image
    
        * contribution
            ➔ SOTA performance without using 'negative sets'.
            ➔ prevent collapse by employinhg soft target update.
            ➔ More resilience to augmentation methods compared to contrastive learning (previous SOTA)
        