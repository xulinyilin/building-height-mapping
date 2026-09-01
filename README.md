# Indochinese Peninsula Building-Height Mapping

Code for annual 5 m building-height mapping across the Indochinese Peninsula,
2016–2023. The complete workflow has four steps.

## Code structure

```text
config/
  sample_preparation.example.yaml
  network_training.example.yaml
gee/
  02_export_training_patches.js
  03_apply_trained_cnn.js
  04_random_forest_height_mapping.js
python/
  01_sample_preparation/prepare_training_samples.py
  02_network_training/convert_gee_tfrecords.py
  02_network_training/model.py
  02_network_training/train_and_export_features.py
```

## Installation

```bash
conda create -n building-height python=3.11 -y
conda activate building-height
pip install \
  "geopandas>=0.14,<2" "numpy>=1.26,<3" "pandas>=2.1,<3" \
  "pyproj>=3.6,<4" "PyYAML>=6,<7" "rasterio>=1.3,<2" \
  "shapely>=2.0,<3" "torch>=2.2,<3" "tfrecord>=1.14,<2"
```

## 1. Obtain building-height labels

`prepare_training_samples.py` performs GEDI quality control and filters
samples using 3D-GloBFP, G-3DUE, ICPIS and Planet temporal stability.

```text
RD = abs((H1 - H2) / H2)
```

```bash
cp config/sample_preparation.example.yaml config/sample_preparation.yaml
python python/01_sample_preparation/prepare_training_samples.py \
  --config config/sample_preparation.yaml
```

## 2. Extract image patches in GEE

Run `gee/02_export_training_patches.js` in the Earth Engine Code Editor. It:

- splits labels into 80% training and 20% validation samples;
- filters samples by their corresponding year;
- creates annual median Planet NICFI and annual mean Sentinel-1 composites;
- extracts 35 × 35 patches in R, G, B, NIR, VV and VH order;
- exports the batches to Google Drive as compressed TFRecord files.

Convert the exported TFRecords to the NPY and JSON inputs used by the local
training code:

```bash
python python/02_network_training/convert_gee_tfrecords.py \
  --inputs "data/raw/train/*.tfrecord.gz" \
  --output-data data/processed/Samples_Train_data.npy \
  --output-properties data/processed/Samples_Train_properties.json

python python/02_network_training/convert_gee_tfrecords.py \
  --inputs "data/raw/val/*.tfrecord.gz" \
  --output-data data/processed/Samples_Val_data.npy \
  --output-properties data/processed/Samples_Val_properties.json
```

## 3. Train the dual-branch CNN locally

The Planet and Sentinel-1 branches each generate 16 final-convolution
features, producing 32 deep features in total.

```bash
cp config/network_training.example.yaml config/network_training.yaml
python python/02_network_training/train_and_export_features.py \
  --config config/network_training.yaml
```

The program exports the trained model, evaluation results, 32 deep features
and Earth Engine convolution kernels.

## 4. Perform random-forest height inversion

First run `gee/03_apply_trained_cnn.js` to generate the 32 CNN feature bands.
Then run `gee/04_random_forest_height_mapping.js` to combine:

- 22 Planet features;
- 19 Sentinel-1 features;
- 18 Sentinel-2 features.

The random forest uses 100 trees and a minimum leaf size of 5. Sentinel-2
clouds are masked with Cloud Score+ `cs_cdf`, and its spectral bands are
multiplied by `0.0001` before index calculation. The final annual
building-height maps are masked using ICPIS and exported at 5 m resolution.

## Notes

- Replace all example local paths and Earth Engine asset identifiers.
- Raw or licensed imagery and large derived products are not distributed.
- Keep the YAML configuration templates because the Python scripts require
  them.

## License

See `LICENSE`.
