# Computational Modeling and Structural Generation of Piano Music in the Classical Style

We present a novel framework for generating coherent piano music in the style of Classical sonatas under low-resource conditions, with particular emphasis on extended duration, rhythmic stability, and well-defined structural organization.

The framework operates on symbolic music token representations—a notation-based format that encodes sequences of discrete note events. By incorporating sonata structure via control tokens that specify phrase types, the framework captures both phrase-level organization and long-range dependencies spanning tens of thousands of tokens, despite the model’s 512-token input limit.

## Key Contributions

1. **AutoStruc**  
   A multi-modal automatic structural annotation framework that integrates score, audio, and symbolic representations to construct hierarchical structural annotations from collected sonatas in an unsupervised manner.

2. **Music Generation Framework**  
   - A computational model of the generative process for sonata-style music  
   - **REMI-Lite Tokenization**: a symbolic music tokenization scheme emphasizing beat-bar-phrase structure  
   - **nextGEN** and **accGEN**: two transformer-based, paired-sequence models that conditionally generate melody and accompaniment phrases. These phrases are assembled into structurally coherent sections or full movements.

3. **Evaluation Strategies**  
   A set of methods for assessing both model correctness and musicality, including rhythmic regularity and structural coherence. This includes objective metrics as well as behavioral tests.

---

## Data Pipeline

Prepare the dataset by following the steps in [`xxREADME.md`](xxREADME.md).

---

## Training

```bash
python3 ./src/train_nextGEN.py
python3 ./src/train_accGEN.py
```

---

## Inference
```bash
python3 ./src/inference.py
```

---

## Evaluation

---

## Sample Output
Listen to samples in https://yijingf.github.io/Sonata-Struct-GEN/



