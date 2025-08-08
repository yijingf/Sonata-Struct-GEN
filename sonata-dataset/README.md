# Sonata Dataset

We curated the **Sonata Dataset** comprising Classical piano and harpsichord sonatas by four composers, [Scarlatti](https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/scarlatti/longo), [Haydn](https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/haydn/keyboard/uesonatas), [Mozart](https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/mozart/piano/sonata), and [Beethoven](https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/beethoven/piano/sonata). To ensure stylistic consistency, only Beethoven’s sonatas composed before 1810 were included.

The number of movements contributed by each composer
|  Composer | # of Movements |
|:---------:|:--------------:|
| Scarlatti |       65       |
|   Haydn   |       25       |
|   Mozart  |       69       |
| Beethoven |       85       |

The Sonata Dataset consists of 244 movements from 126 sonatas. These sonatas are presented in three modalities:
* its original form as text-based digital scores in **kern data format obtained from [KernScores](https://kern.ccarh.org/) library,
* MIDI files converted from the **kern scores,
* audio files rendered from MIDI.

## Kern Score Collection
1. Download `.krn` files for each composer as a zipped archive, and place them under:
    ```
        ./krn/<composer>
    ```
    Replace `<composer>` with one of: scarlatti, haydn, mozart, or beethoven.
2. Use `get_meta.py` to generate MIDI files and metadata:
    * MIDI files will be saved in `./downloaded_midi/<composer>`
    * Metadata will be saved as CSV in `./info/<composer>.csv`
3. Typos and minor errors in `.krn` files were manually corrected to enable proper parsing. See [Appendix](Appendix.md) for more details.

## Dataset Layout
```
sonata-dataset/
├── info/<composer>.csv
│
├── krn/<composers> # original scores stored as krn files
│
├── downloaded_midi/<composers> # original downloaded midi
│
├── mxml/<composers>/ # parsed score stored as  files
│
├── event/<composer>/ # parsed score stored as JSON files
│
├── midi/<composers> # midi rendered from JSON files in ./event and change point JSON files
│
├── dataset_split.csv # train/val/test split
│
├── event_part/ # 
│   ├── acc/<composers>    # accompaniment event
│   ├── melody/<composers>    # melody event
│   ├── norm_acc/<composers>    # normalized and expanded accompaniment
│   └── norm_melody/<composers>    # normalized and expanded melody
│ 
├── struct_mark/ # produced by AutoStruc
│ 
├── dataset/ # 
│   ├── nextGEN_pretrain/    # 
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json    # normalized and expanded melody
│   │   
│   ├── nextGEN_finetune/    # accompaniment event
│   │   ├── ...
│   │
│   ├── accGEN/    # accompaniment event
│   │   ├── ...
│
├── scripts/              # command‐line entry points/wrappers
│   └── *.py
└── README.md
```

## Data Process Pipeline
See `src/data_pipeline/README.md` for more details.
