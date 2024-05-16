import os
import bidsio
import bids
import monai
from finetune import ATLASPredictor

if __name__ == '__main__':
    output_dir = 'predictions/'
    os.makedirs(output_dir, exist_ok=True)

    bids_loader = bidsio.BIDSLoader(data_entities=[{'subject': '',
                                                    'session': '',
                                                    'suffix': 'T1w',
                                                    'space': 'MNI152NLin2009aSym'}],
                                    target_entities=[],
                                    data_derivatives_names=['ATLAS'],
                                    batch_size=4,
                                    root_dir='data/test/')

    model_path = 'models/finetune_best_val_loss.pt'
    base_model = monai.networks.nets.SwinUNETR(img_size=[96, 96, 96],
                                          in_channels=1,
                                          out_channels=1,
                                          feature_size=48,
                                          use_checkpoint=True,
                                          depths=[2, 2, 2, 2],
                                          num_heads=[3, 6, 12, 14],
                                          drop_rate=0.1)

    model = ATLASPredictor(base_model, out_size=[19])


    for data, image_list in bids_loader.load_batch_for_prediction():
        prediction =