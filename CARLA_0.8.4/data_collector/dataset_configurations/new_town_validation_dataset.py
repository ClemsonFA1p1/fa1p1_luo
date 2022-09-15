from carla import sensor
from carla.settings import CarlaSettings
"""
Configuration file used to collect the CARLA 100 data.
A more simple comment example can be found at coil_training_dataset_singlecamera.py
"""
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
POSITIONS =  [[19, 66], [79, 14], [19, 57], [39, 53], [60, 26],
             [53, 76], [42, 13], [31, 71], [59, 35], [47, 16],
             [10, 61], [66, 3], [20, 79], [14, 56], [26, 69],
             [79, 19], [2, 29], [16, 14], [5, 57], [77, 68],
             [70, 73], [46, 67], [34, 77], [61, 49], [21, 12]]

FOV = 100
sensors_frequency = {'CentralRGB': 1, 'CentralDepth': 1, 'CentralSemanticSeg':  1, 'Lidar32': 1,
                     'RightRGB': 1, 'RightDepth': 1, 'RightSemanticSeg': 1,
                     'LeftRGB': 1, 'LeftDepth': 1, 'LeftSemanticSeg': 1}
sensors_yaw = {'CentralRGB': 0, 'CentralDepth': 0,'CentralSemanticSeg': 0, 'Lidar32': 0,
                 'RightRGB': 30.0, 'RightDepth': 30.0, 'RightSemanticSeg': 30.0,
                 'LeftRGB': -30.0, 'LeftDepth': -30.0, 'LeftSemanticSeg': -30.0}
lat_noise_percent = 20
long_noise_percent = 20
NumberOfVehicles = [30, 60]  # The range for the random numbers that are going to be generated
NumberOfPedestrians = [50, 100]
set_of_weathers = [1, 3,6,8]

def make_carla_settings():
    """Make a CarlaSettings object with the settings we need."""

    settings = CarlaSettings()
    settings.set(
        SendNonPlayerAgentsInfo=True,
        SynchronousMode=True,
        NumberOfVehicles=30,
        NumberOfPedestrians=50,
        WeatherId=1)

    settings.set(DisableTwoWheeledVehicles=True)

    settings.randomize_seeds() # IMPORTANT TO RANDOMIZE THE SEEDS EVERY TIME

    camera0 = sensor.Camera('CentralRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 0, 0)

    settings.add_sensor(camera0)
    
    camera0 = sensor.Camera('LeftRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, -1.3, 1.4)
    camera0.set_rotation(-15.0, -15.0, 0)

    settings.add_sensor(camera0)

    # settings.add_sensor(camera0)
    # camera0 = sensor.Camera('RightRGB')
    # camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    # camera0.set(FOV=FOV)
    # camera0.set_position(2.0, 1.3, 1.4)
    # camera0.set_rotation(-15.0, 0.0, 0)

    #settings.add_sensor(camera0)

    # lidar = sensor.Lidar('Lidar32')
    # lidar.set_position(2, 0, 1.4)
    # lidar.set_rotation(0, 0, 0)
    # lidar.set(
    #     Channels=32,
    #     Range=50,
    #     PointsPerSecond=100000,
    #     RotationFrequency=10,
    #     UpperFovLimit=10,
    #     LowerFovLimit=-30)
    # settings.add_sensor(lidar)
    return settings
