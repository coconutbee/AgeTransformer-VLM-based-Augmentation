from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path , transform, resolution=128):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
        
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            label_key = f'{self.resolution}-label-{str(index).zfill(5)}'.encode('utf-8')
            label = txn.get(label_key)
            label2_key = f'{self.resolution}-label2-{str(index).zfill(5)}'.encode('utf-8')
            label2 = txn.get(label2_key)
            name_key = f'{self.resolution}-name-{str(index).zfill(5)}'.encode('utf-8')
            name = txn.get(name_key)
            

        buffer = BytesIO(img_bytes)
        # print('buffer',buffer)
        # print('namekey',name)
        # name = name.split('/')[-1]
        img = Image.open(buffer)
        img = self.transform(img)

        return img, int(label)
