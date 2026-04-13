# Data

Raw and processed GTSRB files live here. The contents of `raw/` and
`processed/` are gitignored; only this README is checked in.

## Layout

```
data/
├── README.md                 # this file
├── raw/
│   └── gtsrb_4gb/            # nine preprocessed pickles from Kaggle
│       ├── data0.pickle
│       ├── ...
│       ├── data8.pickle
│       └── label_names.csv
└── processed/
    └── data0images/          # PNGs exploded by the preprocessing step
        ├── data0train/<label>/image_*.png
        ├── data0validation/<label>/image_*.png
        └── data0test/<label>/image_*.png
```

The default data root can be overridden with `DSD_DATA_ROOT=/some/path`.

## Downloading the 4GB preprocessed bundle

Source: [`valentynsichkar/traffic-signs-preprocessed`](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed) on Kaggle (~4GB).

1. Install the project deps — this pulls in the `kaggle` CLI:

   ```bash
   uv sync
   ```

2. Create a Kaggle API token:

   - Go to <https://www.kaggle.com/settings>, scroll to **API**, click **Create New Token**.
   - Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json` and lock it down:

     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. Run the downloader:

   ```bash
   uv run python scripts/download_data.py
   ```

   The bundle is written to `data/raw/gtsrb_4gb/` and auto-unzipped.

## Extracting images for FastAI

FastAI's `DataBlock` reads files from disk, so the pickles need to be exploded
into `label/image_*.png` trees:

```bash
uv run python -m dashcam_sign_detector.classifier.preprocessing --data-id 0
```

Pick `--data-id 0..8` depending on which preprocessing variant you want:

| id | notes |
|---:|-------|
| 0  | Shuffling, RGB |
| 1  | + `/255.0` normalization |
| 2  | + mean normalization |
| 3  | + mean + std normalization |
| 4  | Grayscale, shuffled |
| 5  | + local histogram equalization |
| 6  | + `/255.0` normalization |
| 7  | + mean normalization |
| 8  | + mean + std normalization |

The baseline reproduction uses `--data-id 0` with ResNet-34.

## License

GTSRB is commonly used for research/education; verify its license
([original site](https://benchmark.ini.rub.de/gtsrb_news.html)) before
redistributing derived artifacts. Cite the dataset as:

> J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. *The German Traffic
> Sign Recognition Benchmark: A multi-class classification competition*.
> IJCNN 2011.
