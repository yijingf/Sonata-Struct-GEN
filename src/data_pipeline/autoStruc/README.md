# Structural Segmentation

We extract the structural graph of sonata music through a combination of **phrase segmentation** and **music form analysis**.

Phrase boundaries are detected from audio in an unsupervised manner by grouping similar patches of audio descriptors, enhanced with a triplet mining method. 

Music form analysis is rule-based. We estimate section boundaries indicated by score notations (such as repeat signs, key changes, and time signature changes), and then refine the phrase boundaries detected from audio.

## Segmentation Heuristics

The capital letter notations in `.krn` files represent materials between two repeat signs, referred to as **patterns** of **score sections**. Most movements—except those in air variations or simple binary forms—follow a three-section structure. Typically, the third section begins with the second or third entry of the initial theme introduced in the first section.

We use the following heuristics to estimate boundaries of larger **sections**, depending on the pattern types:

- **A**: Section boundaries are identified by occurrences of elements identical to the first phrase. If no repeated material is detected, we assume it is a single-section movement.

- **AB**: We assume either a recapitulation is present in **B**, or no recapitulation exists. The development starts from **B**.

- **XYZ**: Patterns with more than three score sections—such as **ABCD**, **ABCDE**, etc.—belong to this type. The recapitulation is assumed to be the second or third entry of the primary theme after **B**, or it may be absent.

- **XYX**: The second **X** is an exact repeat or a variation of the first **X**. This pattern is typically presented as **AB - x - AB** or **AB - x - ABy**, where **x** is different from **AB**. In this case, the first **B** marks the end of the exposition, and the second **A** marks the start of the recapitulation.

- **I-x**: If the score section notation starts with **I**, it is assumed to be an **Introduction**. The segmentation rules above apply to the remaining material.
