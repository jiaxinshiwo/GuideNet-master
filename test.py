import torch

if __name__ == "__main__":
    # only for test
    from datasets.cholecdata import CholecData, make_pretrain_dataset
    from datasets.cholecdata import make_guide_dataset
    from utils.spatial_transforms import (Compose, Scale, ToTensor)
    from utils.target_transforms import (FlowLabel, ClassLabel)
    from utils.mylogger import setup_logger
    import torch.utils.data as data

    # setup_logger()
    root_path = '/home/amax/lab_data/surgicalData/cholec80_data/remain_time/remain_time/'
    anna_path = '/home/amax/lab_data/surgicalData/cholec80_data/data.pkl'

    # root_path = '/home/amax/lab_data/surgicalData/MICCAI_2019/Frames/Full/'
    # anna_path = '/home/amax/lab_data/surgicalData/MICCAI_2019/Annotation/data.pkl'

    dataset = CholecData(root_path=root_path, n_samples_for_each_video=8, sample_duration=10,
                         annotation_path=anna_path, spatial_transform=Compose([Scale((224, 224)), ToTensor()]),
                         target_transform=None, subset='train', is_finetune=False, is_guide=False)

    dataloader=data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    for i, (inputs, targets) in enumerate(dataloader):
        tel_labels = targets[:, 0]
        time_labels = targets[:, 1]
    # 计算进度以及剩余时间
        regression_labels=[torch.true_divide(tel_labels[i],time_labels[i]) for i in range(len(tel_labels))]
        rsd_labels = torch.true_divide((time_labels - tel_labels),7500)
        print(rsd_labels)
    # data=make_pretrain_dataset(root_path, anna_path, subset='all', n_sample_for_each_video=10, sample_duration=15)
    # for i in data:
    #     print(data)
    #     print("**************************")