# FA1p1_Luo

Repository for Dr. Luo


# Adding quick commands for now. Will make nicer later on

#Run baseline driving network
Check that the config file for the assocated folder and configuration has the IMAGE_TRANSLATION config set to False

python coiltraine.py --gpus 0  --folder nocrash --exp resnet34imnet10S1 --single-process drive -de NocrashTraining_Town01 --docker carlagear 

# Run translation
Check that the config file for the assocated folder and configuration has the IMAGE_TRANSLATION config set to True
Choose translation checkpoint via the -name and --which_epoch parameters.

python coiltraine.py --gpus 0  --folder nocrash --exp resnet34imnet10S1 --single-process drive -de NocrashTraining_Town01 --docker carlagear --name finetune_fromEpoch400_episodes_1000epoch_weight2000.0 --which_epoch 200 --no_instance --n_downsample_global 2

