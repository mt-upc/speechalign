# SpeechAlign: a Framework for Speech Translation Alignment Evaluation

[\[Paper\]](https://arxiv.org/abs/2309.11585)


Download the original dataset (Vilar et al., 2006) from [here](https://www-i6.informatik.rwth-aachen.de/goldAlignment/)


>David Vilar, Jia Xu, Luis Fernando D’Haro, and Hermann Ney. 2006. Error Analysis of Statistical Machine Translation Output. In Proceedings of the Fifth International Conference on Language Resources and Evaluation (LREC’06), Genoa, Italy. European Language Resources Association (ELRA).


## Generate the dataset

To be able to use the dataset, you will have to generate it by yourself. Here you will find the instructions to do so.

### Docker

#### CPU

```bash
docker run -it \
    -v $(pwd):/home/sga \
    --workdir /home/sga \
    --entrypoint /bin/bash \
    ghcr.io/coqui-ai/tts-cpu:e5fb0d96279af9dc620add6c2e69992c8abd7f24 \
    ./generate_dataset.sh
```

#### GPU

```bash
docker run -it --gpus all \
    -v $(pwd):/home/sga \
    --workdir /home/sga \
    --entrypoint /bin/bash \
    ghcr.io/coqui-ai/tts:e5fb0d96279af9dc620add6c2e69992c8abd7f24 \
    ./generate_dataset.sh
```
