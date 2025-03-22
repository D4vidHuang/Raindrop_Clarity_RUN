# Raindrop_Clarity_RUN
This is the submission of runnable file for the group Dfusion from TUDelft, with student Yongcheng Huang(MSc DSAIT), Xingyu Han(MSc DSAIT) and Anyan Huang (MSc DSAIT). 

# How to run the code
1. You need to prepare the following pretained model from:
    - DiT Day: https://www.dropbox.com/scl/fo/oy0s69m4jienlvjpu52wi/AP-W58fUHkQm6XVEcMtfPCE/RainDrop/DIT?preview=RainDrop_DiT_ddpm.pth.tar&rlkey=jn0wkbaf8d4xv8rqixhhuhymy&subfolder_nav_tracking=1&st=csviulk7&dl=0
    - DiT Night: https://www.dropbox.com/scl/fo/oy0s69m4jienlvjpu52wi/ANp_bbG4H_GxEoD49-dYF1Y/NightRaindrop/DIT?preview=NightRaindrop_DiT_ddpm.pth.tar&rlkey=jn0wkbaf8d4xv8rqixhhuhymy&subfolder_nav_tracking=1&st=745iymb9&dl=0
    - RDiffusion Day：https://www.dropbox.com/scl/fo/oy0s69m4jienlvjpu52wi/ANFVxYPONv_Iueu5STDHZK4/RainDrop/RDiffusion?preview=RainDrop_ddpm.pth.tar&rlkey=jn0wkbaf8d4xv8rqixhhuhymy&subfolder_nav_tracking=1&st=e8rey2rm&dl=0
    - RDiffusion Night: https://www.dropbox.com/scl/fo/oy0s69m4jienlvjpu52wi/AMTfMKLTSI7H_HpQAx5KPeE/NightRaindrop/RDiffusion?preview=NightRaindrop_ddpm.pth.tar&rlkey=jn0wkbaf8d4xv8rqixhhuhymy&subfolder_nav_tracking=1&st=pu8wr1tm&dl=0
    
2. After you downloaded all pretrained weights, please go to `makeDataset.py` and set the direction to the direction of your actual data direction.
3. Feed the results of the `makeDataset.py` to the models and get their results. You can use the code from paper Raindrop Clarity, in the following link: https://github.com/jinyeying/RaindropClarity?tab=readme-ov-file
4. Please go to `restoration.py` to recompose the results from the SOTA models to 720*480 images. 
5. After you get all complete results from the current SOTA models, please set the parameters in `train.py` to specify the training direction. **Notice: You don't need the test set results from SOTA models to train, but you need to use the training set and its ground truth for training.**
6. Go to the `test.py` and set the directions. It will output you the fused results. **Notice: There is a parameter for day/night, this is only for changing direction easier, there's only one model running, so you can put everything in day/night folder for running.**

# From the authors
We hope this method can be used in different areas to enhance the results better. As this is a plug-to-use trick, I hope this can be easier for enhancing different results in different fields.