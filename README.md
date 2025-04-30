# toxic-cloud-segmentation

Prepare the data
1. IJmond_SEG: from roboflow in COCO format
2. IJmond_VID: use the file metadata_ijmond_jan_22_2024.json, first download the videos using the script 
download_videos.py (in this script only positively labelled videos are selected), then use the script 
preprocessing.py to extract all frames from the selected videos.
3. RISE: use the file metadata.json, filter on positive videos and select the first frame.

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