# toxic-cloud-segmentation

Prepare the data
1. IJmond_SEG: from roboflow in COCO format
2. To get the unlabelled data, run the script load_unlabelled_data.py. This will download the videos and save the 
frames in the data folder. To get IJmond_VID the file metadata_ijmond_jan_22_2024.json is used and all frames are saved. 
To get RISE the file metadata.json is used and only the first frame of each video is saved.

Split the IJmond_SEG dataset into train and test sets by running the make_splits.py script.

Train the baseline model by running the shell script train_ssl.sh. If you want to change the hyperparameters for 
training, go to the train.py file directly. Test the model by running the shell script test_ssl.sh.

To run the experiments, change the hyperparameters in the shell scripts and run them. 

Experiments done for this study:
1. Cross entropy loss with pixel-wise contrastive loss
2. Cross entropy loss with local contrastive loss
3. Cross entropy loss with directional contrastive loss
4. Cross entropy loss with hybrid contrastive loss
5. Dice loss with pixel-wise contrastive loss
6. Dice loss with local contrastive loss
7. Dice loss with directional contrastive loss
8. Dice loss with hybrid contrastive loss