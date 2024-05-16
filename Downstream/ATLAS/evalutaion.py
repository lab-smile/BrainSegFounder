import os
import bidsio
import monai
import torch
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

    model = ATLASPredictor(base_model, out_size=[197, 223, 189])

    model.load_state_dict(torch.load(model_path))

    for data, image_list in bids_loader.load_batch_for_prediction():
        prediction = model(data)
        for i in range(prediction.shape[0]):
            pred_out = prediction[i, 0, ...]
            image_ref = image_list[i][0]
            bids_loader.write_image_like(pred_out, image_ref,
                                         new_bids_root=output_dir,
                                         new_entities={
                                             'label': 'L',
                                             'suffix': 'mask'
                                         })

    bids_loader.write_dataset_description(output_dir,
                                          dataset_name='atlas2_prediction',
                                          author_names=['Joseph Cox'])
