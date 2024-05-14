# SpeechAlign: a Framework for Speech Translation Alignment Evaluation

[\[Paper\]](https://arxiv.org/abs/2309.11585)


## Get dataset

You will have to regenerate the dataset by yourself. We provide easy-to-follow instructions on how to do it using containers.

### Download the original dataset

Download the original dataset (Vilar et al., 2006) from [here](https://www-i6.informatik.rwth-aachen.de/goldAlignment/) and unzip it in the `dataset` folder.


>David Vilar, Jia Xu, Luis Fernando D’Haro, and Hermann Ney. 2006. Error Analysis of Statistical Machine Translation Output. In Proceedings of the Fifth International Conference on Language Resources and Evaluation (LREC’06), Genoa, Italy. European Language Resources Association (ELRA).


### Generate the dataset

Use Docker or Apptainer (Singularity) to generate the dataset. 

#### Docker

With GPU:

```bash
docker run -it --gpus all \
    -v $(pwd):/home/sga \
    --workdir /home/sga \
    --entrypoint /bin/bash \
    ghcr.io/coqui-ai/tts:e5fb0d96279af9dc620add6c2e69992c8abd7f24 \
    ./generate_dataset.sh
```

Without GPU (this will take a long time):

```bash
docker run -it \
    -v $(pwd):/home/sga \
    --workdir /home/sga \
    --entrypoint /bin/bash \
    ghcr.io/coqui-ai/tts-cpu:e5fb0d96279af9dc620add6c2e69992c8abd7f24 \
    ./generate_dataset.sh
```


#### Apptainer (aka Singularity)

With GPU:

```bash
apptainer exec --nv \
    docker://ghcr.io/coqui-ai/tts:e5fb0d96279af9dc620add6c2e69992c8abd7f24 \
    bash ./generate_dataset.sh
```

Without GPU (this will take a long time):

```bash
apptainer exec --nv \
    docker://ghcr.io/coqui-ai/tts-cpu:e5fb0d96279af9dc620add6c2e69992c8abd7f24 \
    bash ./generate_dataset.sh
```

### Compute AER
Before obtaining the word-AER (Alignment Error Rate) metric, it is necessary to generate contributions maps for each sentence in the Speech Gold Alignment dataset. These maps should be stored in the .pt format, and each file must be named according to the corresponding sample index in the dataset: {idx}.pt. The sample index should be a three-digit number, such as 001, 011, or 111. The map should not contain contributions for the end of sentence token.

```bash
python3 speech_aer/aer.py  --test_set_dir /path/to/folder/ \ # path to the Speech Gold Alignment dataset folder.
                --path_to_contribs /path/to/folder/ \ # path to the folder with token to token contributions. 
                --path_to_tokenized_targets /path/to/text/file \ # path to txt file with tokenized target sentences
                --save_alignment_hyp /path/to/text/file \ # path to save the alignments hypotesis.
                --setting s2s \ # s2s or s2t 
                --translation_direction en-de \ # en-de or de-en
```

### Visualize contributions and hard-alignments

The notebook ```speech_aer/visualize_alignment.ipynb``` can be used to obtain heatmaps and visualize the word-word contributions and the hard alignments that are used to compute the AER.