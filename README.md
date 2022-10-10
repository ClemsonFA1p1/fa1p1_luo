# FA1p1_Luo

Repository for Dr. Luo

# Contributing
Before adding to this repo it is recommended to set up a .gitignore file and add the pycache folder 
 
# Run baseline driving network
Check that the config file for the assocated folder and configuration has the IMAGE_TRANSLATION config set to False

python coiltraine.py --gpus 0  --folder nocrash --exp resnet34imnet10S1 --single-process drive -de NocrashTraining_Town01 --docker carlagear 

# Run translation
Check that the config file for the assocated folder and configuration has the IMAGE_TRANSLATION config set to True and the STYLE and AERIAL configs are False.

Choose translation checkpoint via the -name and --which_epoch parameters.

python coiltraine.py --gpus 0  --folder nocrash --exp resnet34imnet10S1 --single-process drive -de NocrashTraining_Town01 --docker carlagear --name finetune_fromEpoch400_episodes_1000epoch_weight2000.0 --which_epoch 200 --no_instance --n_downsample_global 2

# Run translation with styleGan
This is the model which is used for the aerial translation.

Ensure that the configuration file correctly set STYLE_TRANSLATION and AERIAL_TRANSLATION. You may also have to change these files in coil_global.py if they are not correctly adjusted.

Be sure to replace checkpoint path with the desired checkpoint

python coiltraine.py --gpus 0  --folder nocrash --exp resnet34imnet10S1 --single-process drive -de NocrashTraining_Town01 --docker carlagear --checkpoint_path pixel2style2pixel/checkpoints/carla_AtoG/checkpoints/iteration_1000000.pt

# Run data_collector
The data collection must be run under the old translation environment pix2pix

python multi_gpu_collection.py -pt /path/to/data/folder -d dataset_configuration_file


