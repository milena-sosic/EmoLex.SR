# WNA.SR - WordNet-Affect for Serbian WordNet (SWN)

WNA.SR is a Python package designed to integrate WordNet-Affect with Serbian WordNet. This package adds affective categorization to Serbian WordNet.

It is based on the created linkage between the WordNet 1.6 and WordNet 3.0 established by Japaneese scientists, following the code found [here](https://github.com/skozawa/japanese-wordnet-affect?tab=readme-ov-file)

## Features

- **Load WordNet-Affect Synsets**: Load synsets from WordNet-Affect XML files.
- **Merge Synsets**: Merge WordNet-Affect synsets with WordNet 3.0 synsets.
- **Output Serbian WordNet**: Generate Serbian WordNet with affect annotations.

## Requirements

- Python 3.x
- [WordNet-Affect](https://wndomains.fbk.eu/wnaffect.html) as extension of [WordNet-Domains](https://wndomains.fbk.eu/index.html) - provides hierachically organized set of emotional words
- [Princeton WordNet](https://wordnet.princeton.edu/) - versions 1.6 and 3.0 data files
- [Serbian WordNet](https://wn.jerteh.rs/) (SWN)
- XML libraries (ElementTree)
- NLTK
- Pandas



