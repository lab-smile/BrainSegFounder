# GatorBrain pretraining code
## Installation steps
1) Clone repository
2) Install requirements
    ```bash
    $ pip install -r requirements.txt
    ```
3) Change `config.ini` to customize to training needs
   1) **General**: 
      * encoder/decoder - which encoder/decoder used to pretrain (do not change atm)
      * img_type - which type of MRI to pretrain on (can be anything in UKB)
      * small_train - whether to only train on first 1000 images that meet image type or all images 
   2) **Data**:
      * data_dir: this code utilizes the non-public UK Biobank Dataset. There is no download. You need to have the 
        `img_type` MRI data as NIFTI downloaded for pretraining data, in individual patient files
         stored in the directory listed here: e.g. `{data_dir}/{patient_id}/{img_type}.nii.gz`
   3) **Pretraining**: (self explanatory, will update later)
   4) **Transformations**: 
      * Any `{trasformation}_rate` is the probability of that transformation 
        happening to a given image. For image in/out painting - the painting rate
        is the probability of any paint happening, and then the inpainting rate is the 
        probability of an inpaint happening given a paint is already happening. Outpaint
        rate is calculated as `1 - {inpainting_rate}`
      * num_slices is the number of slices (3D-cubes) to split each transformable MRI into
      * window_size is the size of each of the above slices 
   5) **Logging**: (self explanatory)
4) Run the code: 
```bash
$ python GatorBrain.py
```
   An example script opening a conda environment and running the pretraining code can be found in `train.sh`.
