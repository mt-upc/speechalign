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
