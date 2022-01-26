from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, NDArrayField, IntField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder
import numpy as np
from pathlib import Path
from multilabel.config import Config

NDArrayArgsType = np.dtype([
    ('shape', '<u8', 32),  # 32 is the max number of dimensions for numpy
    ('type_length', '<u8'),  # length of the dtype description
])

def get_ffcv_dataloader(dataset,ids,dataset_path):
    Path(Config.ffcv_dataset_path).mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(dataset_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': RGBImageField(max_resolution=256, jpeg_quality=90),
    'label': NDArrayField(dtype = NDArrayArgsType, shape = (Config.num_classes,))
    })
    writer.from_indexed_dataset(dataset)

    decoder = RandomResizedCropRGBImageDecoder((Config.img_size['height'], Config.img_size['width']))

# Data decoding and augmentation
    image_pipeline = [decoder, Cutout(), ToTensor(), ToTorchImage(), ToDevice(0)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(0)]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    loader = Loader(dataset_path, batch_size=Config.batch_size, num_workers=4,
                    order=OrderOption.RANDOM, pipelines=pipelines, indices = ids)

    return loader
