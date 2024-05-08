# Exert Diversity and Mitigate Bias: Domain Generalizable Person Re-identification with a Comprehensive Benchmark

## Benchmark Construction
### Datasets Information
![](./intro.png)

| Datasets                   | Collected Scenes    | \# Identities | \# Cameras | \# Total Images |
|----------------------------|---------------------|---------------|------------|-----------------|
| Small-Scale
| iLIDS                           | Airport Arrival Hall       | 119                 | 2             | 476        |
| GRID                       | Underground Station | 1,025         | 8          | 1,275           |
|PKU                        | Campus              | 114           | 2          | 1,824           |
|SAIVT                      | Buildings           | 152           | 8          | 7,150           |
| Medium-Scale
| Market-1501 | Campus              | 1,501         | 6          | 29,419          |
| LPW                        | Street              | 2,731         | 4          | 30,678          |
| WildTrack                  | An Open Area        | 313           | 7          | 33,979          |
| Airport                    | Airport             | 9,651         | 6          | 39,902          |
| Large-Scale 
| SOMAset    | Synthetic           | 50            | -          | 100,000         |
| Unreal                     | Synthetic           | 1,960         | 34         | 119,128         |
| MSMT17                     | Indoor and Outdoor  | 4,101         | 15         | 126,441         |
|PersonX                    | Synthetic           | 1,266         | 6          | 273,456         |

### Evaluation Procotols
![](./protocol.png)


### Pipeline of $DF^2$
![](./framework.png)


### Requirements
+ CUDA>=10.0
+ Four 1080-Ti GPUs
+ necessary packages listed in [requirements.txt](requirements.txt)


## Acknowledgments
This repo borrows partially from [fast-reid](https://github.com/JDAI-CV/fast-reid)
