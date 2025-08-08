# MANUAL CORRECTIONS
In addition, we manually complete measures at first volta brackets because Music21 is not reliable in estimating measure index after these repeat barlines. It estimates `measure.number` based on the bar line and global `measure.offset`, i.e. beat count at the beginning of the bar, thus is not reliable when measures at volta brackets are incompelete. See [Appendix-Table 2](Appendix.md#table-2) for more details.

### Table 1
| Title | # Bar | Fix | Issue |
|:-----:|:---:|:---:|:---:|
| mozart/sonata04-3 |  | add `*MM120 *MM120  *MM120` in line 19 | Missing tempo |
|  mozart/sonata06-3c |  | add `*MM120 *MM120  *MM120` in line 19 | Missing tempo |
| mozart/sonata06-3i |  | add `*MM120 *MM120  *MM120` in line 19 | Missing tempo |
| mozart/sonata11-1g |  | add `*MM120 *MM120  *MM120` in line 19 | Missing tempo |
| mozart/sonata03-3 | 43 | add `*M8/2 *M8/2   *M8/2` in line 547 | Too many notes; change time signature accordingly |
| scarlatti/L400K360 |  |  `[A,A,B,B]` -> `[A,A1,A,B,B]` | Incorrect repetition pattern |
| scarlatti/L503K513 | 58 | `(16cc 16 cccLL` -> `(16cc 16cccLL` | Typo in notes |
| scarlatti/L013K060 | 25 | add `*>B  *>B *>B` | Missing section name |
| scarlatti/L334K122 | 1 | `=1` -> `=1-` line 19 | Typo in measure delimiter |
| beethoven/sonata14-3 | 188 | remove `*cue` | Typo |
| beethoven/sonata24-1 | 4 | add `*M4/4 *M4/4   *M4/4`; reindex bar 1-4 as 0-3 | Missing time signature |
| mozart/sonata09-3| 173 |break into 1-172, 174-end | extra beats |
| mozart/sonata03-13| 199 | break into 1-198, 200-end | extra beats |
| mozart/sonata14-2| 29, 30, 31, 50, 51, 52 | break into 1-28, 32-49, 53-end | extra beats |
| beethoven/sonata03-1 | 232 | break into 1-232 + 1/2 bar,  233-end | extra beats |
| beethoven/sonata13-4 | 263, 264, 265 | break into 0-263 + 1/2 bar,  266-end | extra beats |
| beethoven/sonata16-2 | 26 | break into 1-26, 27-end | extra beats |
| beethoven/sonata21-3 | 402 | break into 1-402,  402-end | extra beats |
| beethoven/sonata24-2 | 177 | break into 1-176, 178-end | extra beats |
| hayden/sonata42-3 | 57 | remove  `8r	8r	.` in line 362, reindex measure 57-73 by -1 | extra beats |
| haydn/sonata59-1 | 131 | remove notes between `cue` and `Xcue`; change tempo to `3/4` | extra beats |
| beethoven/sonata04-3 | 96 | add `2r	2r	.`| missing beats |
| beethoven/sonata04-4 | 65 | move `*>B	*>B	*>B` inside measure 65 | typo |
| beethoven/sonata18-3 | 20 | `==20` -> `=20`| typo |
| beethoven/sonata18-3 | 27 | `8r	8r	.` to `4r	4r	.`, reindex measure by -1 starting from measure 28 | incomplete measure |
| beethoven/sonata18-3 | 60 | `8r	8r	8r	.` to `4r	4r	4r	.` in line 462 | incomplete measure |
| beethoven/sonata18-4 | | move `*>A	*>A	*>A` inside measure 1 | typo |
| scarlatti/L481K025| | change all tempo to 112 | Ritardando at the end of a section|

### Table 2
| Title | # Bar | Fix | NOTE |
|:-----:|:---:|:---:|:---:|
| mozart/sonata11-3 | 96 | add `4r . 4r    .` | music21 parses the added rest as `C4` in measure 96, <br>which needs to be removed manually \| |
| haydn/sonata34-1 | 68 | `8r 8r  .` -> `4r   4r  .` |  |
| beethoven/sonata01-4 | 57 | add `2r 2r  .` | increase index by 1 after measure |
| beethoven/sonata02-1 | 117 | `8r 8r  .` -> `4r   4r  .` |  |
| beethoven/sonata03-3 | 64 | `8r  8r  .` -> `2r   2r  .` |  |
| beethoven/sonata03-3 | 107 | `8r  8r  .` -> `2r   2r  .` |  |
| beethoven/sonata06-1 | 66 | `8r  8r  .` -> `4r   4r  .` |  |
| beethoven/sonata12-2 | 91 | `4r  4r  4r  4r  .` -> `2r   2r  2r  2r  .` |  |
| beethoven/sonata13-2 | 79 | `4r  4r  .` -> `2r   2r  .` |  |
| beethoven/sonata18-3 | 59 | `4r  4r  4r  .` -> `4.r  4.r 4.r .` |  |

beethoven sonata03-3 16-2 manually cleaned (incomplete first volta, Nov. 25)