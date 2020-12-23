if __name__ == "__main__":
    # only for test
    from datasets.cholecdata import CholecData

    from utils.spatial_transforms import (Compose, Scale, ToTensor)
    from utils.target_transforms import (FlowLabel, ClassLabel)
    from utils.mylogger import setup_logger

    # setup_logger()
    root_path = '/home/amax/lab_data/surgicalData/cholec80_data/remain_time/remain_time'
    anna_path = '/home/amax/lab_data/surgicalData/cholec80_data/data.pkl'
    dataset = CholecData(root_path=root_path, n_samples_for_each_video=8, sample_duration=20,
                         annotation_path=anna_path, spatial_transform=Compose([Scale((224, 224)), ToTensor()]),
                         target_transform=FlowLabel(), subset='train', is_finetune=True, is_guide=False)

    for i, (inputs, targets) in enumerate(dataset):
        print(targets)