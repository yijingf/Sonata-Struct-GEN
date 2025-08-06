# Sonata Dataset

We curated the **Sonata Dataset** comprising Classical piano and harpsichord sonatas by four composers, [Scarlatti](https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/scarlatti/longo), [Haydn](https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/haydn/keyboard/uesonatas), [Mozart](https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/mozart/piano/sonata), and [Beethoven](https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/beethoven/piano/sonata). To ensure stylistic consistency, only Beethoven’s sonatas composed before 1810 were included.

## Dataset Summary
|  Composer | # of Movements |
|:---------:|:--------------:|
| Scarlatti |       65       |
|   Haydn   |       25       |
|   Mozart  |       69       |
| Beethoven |       85       |

The Sonata Dataset consists of **244 movements from 126 sonatas**. Each sonata is presented in three modalities:


- **Kern format**: text-based symbolic scores obtained from the [KernScores](https://kern.ccarh.org/) library  
- **MIDI**: converted from the `.krn` scores  
- **Audio**: rendered from MIDI  

---

## Kern Score Collection

1. Download `.krn` files for each composer as a zipped archive and place them under:

    ```bash
    sonata-dataset/krn/<composer>
    ```

    Replace `<composer>` with one of: `scarlatti`, `haydn`, `mozart`, or `beethoven`.

2. Use `src/data_crawling/get_krn_meta.py` to download MIDI files and extract metadata:
    - MIDI files will be saved in `./downloaded_midi/<composer>`
    - Metadata will be saved as CSV in `./info/<composer>.csv`

3. Minor errors in `.krn` files were manually corrected to enable proper parsing. See [Appendix](../../sonata-dataset/Appendix.md) for details.

---

## Data Process Pipeline

Run each step **individually**, as some require **manual checks or adjustments**.

---

### Step 1: Parse Scores

Extract music events measure-by-measure from `.krn` files and store them in JSON in `./event/<composer>`. [humextra](https://github.com/craigsapp/humextra) is required for this step.

```bash
# Convert `.krn` to `.xml`, which is more compatible with `music21`
./krn2xml.sh ../ 
python3 parse_score.py
```

To check the parsed scores, listen to the MIDI or audio files rendered from scores.

```bash
python3 render_event.py
```

---

### Step 2: Separate and Normalize Melody and Accompaniment

```bash
# Separate Melody and Accompaniment
python3 separate_score.py

# Normalize melody and accompaniment
python3 normalize_events.py
```

---

### Step 3: Unsupervised Structural Annotation

Before running the unsupervised structural annotation, first run phrase segmentation on audio rendered from score, and place the boundary prediction results under `sonata-dataset/boundary_predictions`. 

See `README.md` in [PhraseSegmentation](https://github.com/yijingf/Phrase-Segmentation) for more details. 

Phrase segmentation output are stored as `.pkl` files containing pairs of timing of predicted boundaries and the cluster indices corresponding to the intervals, for example: 
    
```
    Segment Boundaries: 
    array([[  0.        ,   4.99229025],
           [  4.99229025,   7.72063492],
           [  7.72063492,  10.90176871],
           [ 10.90176871,  20.44517007],
           [ 20.44517007,  25.44907029]])

    Segment class: [ 8, 12,  1,  4, 12]
```

Then run: 

```bash
python3 ./structrual_segmentation/struct_segment.py
```
Get more details on structural segmentation in [REAME for structural segmentation](./structural_segmentation/README.md).

---

### Step 4: Generate Dataset

```bash
# Generate train/validation splits
python3 generate_data_split.py --train_ratio 0.8

# Make melody dataset for pretraining nextGEN
python3 nextGEN_pretrain_dataset.py --measure_len 40 --seq_len 512 --n_hop 2 --n_overlap 2

# Make melody dataset for finetuning nextGEN
python3 nextGEN_finetune_dataset.py --measure_len 40 --seq_len 512 --n_hop 2 --n_overlap 2

# Make dataset for training accGEN
python3 accGEN_dataset.py --measure_len 40 --seq_len 512 --n_hop 2
``` 
