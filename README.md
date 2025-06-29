![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-FF6F00?style=for-the-badge&logo=openai&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)


![Python](https://img.shields.io/badge/Python-3.12-brightgreen)
![ChatGPT](https://img.shields.io/badge/ChatGPT-v3.5-informational)
![GPT](https://img.shields.io/badge/GPT-v4.1-informational)


# EmoLex.SR - Emotion Affect Lexicon for the Serbian Language

`Emolex.SR` is a comprehensive emotion affect lexicon specifically designed for the Serbian language, based on the methodology detailed in our accompanying paper. This repository provides all necessary files, scripts, and guidelines to use and extend the dictionary effectively.


## Technology and Resources Utilization

This project leverages advanced technologies alongside established language resources to innovate and develop new linguistic tools. Key aspects include:

- **Integration of Established Resources**: By utilizing existing language resources such as:

- [NRC EmoLex](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) - Used for aligning emotion categories. In particular, version of this lexicon, called [EmoLex.EN](https://dataverse.fiu.edu/dataset.xhtml?persistentId=doi:10.34703/gzx1-9v95/PO3YGX) with corrected bias and assigned PoS tags was utilized through this research, 
- [WordNet-Affect](https://wndomains.fbk.eu/wnaffect.html),
- [Serbian WordNet](https://wn.jerteh.rs/) - Utilized for compiling a list of synonymous Serbian words across the contexts, 
- [SrpMD4Tagging](https://live.european-language-grid.eu/catalogue/lcr/9294) - Serbian morphological dictionaries for (word, PoS) tagging

- **Expert Knowledge Integration**: The lexicon incorporates expert knowledge from researchers and linguists to ensure accuracy and relevance. This expert input is crucial for validating the automated processes and ensuring the lexicon's reliability.

- **Machine Learning and NLP Techniques**: Utilization of advanced machine learning algorithms and natural language processing techniques to automate the generation of emotional affect labels and to validate the lexicon entries.

- **Data Collection and Curation**: Rigorous data collection and curation processes to ensure the lexicon's accuracy and relevance to the Serbian language context.

- **Validation and Quality Assurance**: Robust validation processes to ensure the accuracy and reliability of the lexicon entries.
  
- **Advanced Computational Methods**: The construction and validation of the emotion-affect lexicon employ cutting-edge computational techniques, including natural language processing (NLP) and automated semantic analysis by LLMs ([ChatGPT](https://openai.com/index/chatgpt/), [GPT-4.1](https://openai.com/index/gpt-4-1/), [XLM-Emo](https://huggingface.co/MilaNLProc/xlm-emo-t), [Multilingual Sentence Transformers](https://github.com/UKPLab/sentence-transformers/blob/master/examples/sentence_transformer/training/multilingual/README.md)), to ensure accuracy and efficiency.

- **Innovative Lexicon Development**: Through the strategic combination of technological advancements and robust lexical databases, `Emolex.SR` presents a novel affective dictionary that meets contemporary linguistic analysis demands.


## Approach

The `Emolex.SR` lexicon was developed following a multi-step approach:
1. **Emotional Words Collection**: Compilation of Serbian words and corresponding English lexicons with affective dimensions.
2. **Translation**: Manual and automated translation (Google Translate, GhatGPT, GPT-4.1) of emotion affect (word, PoS) entries from English lexicon.
3. **Generation**: Automated generation of (lemma, PoS) synonyms and emotion affect labels.
4. **Annotation**: Manual and automated annotation of emotion affect labels for (lemma, PoS) in Serbian.
5. **Validation**: Validation through cross-referencing (NRC.EmoLex, SWN, WNA.SR) and expert verification.
6. **Construction**: Building a JSON-structured, searchable lexicons in two versions:
   - **[EmoLex.SR-v1](#)** - Translated, verified and adapted version of the English lexicon.
   - **[EmoLex.SR-v2](#)** - Expanded version of the lexicon with Serbian emotional affect words.

The lexicon captures affective dimensions and aligns with the methodologies reported in the associated research paper.


## Repository Structure

- **data/**
  - `NRC.EmoLex.EN`: The lexicon file containing emotion affect data for English words.
  - `serbian_emo_affect_words.tsv`: A list of Serbian words utilized in the expansion of the lexicon.

- **lexicons/**
  - Contains all versions of the lexicon that were developed during the translation, adaptation, and validation steps. If you want to reproduce the steps of lexicon construction, please use the links above to collect all necessary resources. Lexicon constructed through this study (v-1/2) could be found on the [ELG](#) repository.
 
- **WNA.SR/**
  - This folder contains resources related to the WordNet-Affect lexical database, which extends WordNet with affective concepts and is specifically linked to the Serbian WordNet.
  
- **construction/**
  - `translation.py`: Script to facilitate the translation of the emotion affect lexicon.
  - `adaptation.py`: Script used to adapt the lexicon upon manual validation on contextual and linguistics changes (output: **EmoLex.SR-v1**).
  - `expansion.py`:  Script used to enrich the EmoLex.SR-v1 lexicon with the broad synonyms and manually crafted emotional affect words in Serbian (output: **EmoLex.SR-v2**).
  - `analyse_emotion.py`: A script for analyzing emotional signals in Serbian text using the Emolex.SR-v2 lexicon.

- **validation/**
  - Set of scripts used to validate particular steps during lexicon construction.

  
## How to Use

### Prerequisites

- Python 3.12
- Required packages listed in `requirements.txt`.

### Running the Scripts

To construct or validate the emotion affect dictionary please ensure that you collected all necessary resouces.


## Plans for Future Improvements
- **Expansion of Entries:** Include a larger corpus of Serbian words and phrases (MWE) to enhance the breadth of affective dimensions captured.
- **Improved Automation:** Develop more sophisticated machine learning models for automated emotion annotation of texts in Serbian.
- **Language Variants:** Extend the dictionary to support additional language dialects and variants (ijekavian).
- **User Interface:** Create a user-friendly graphical interface for easier interaction and analysis.
- **API Development:** Build a REST API for remote and programmatic access to the lexicon.


## License
- MIT General Public License


## Contact
- For any question, please contact us at jerteh.rs@gmail.com.




