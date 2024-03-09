## Speech Enhancement Pipeline

On the extended dataset we apply a speech enhancement pipeline to the audio files. The pipeline consists of the following steps:
- Voice Activity Detection (VAD)
- Dereverberation
- Speech Enhancement

Each step can be applied independently.

### Voice Activity Detection (VAD)

@TODO

### Dereverberation

@TODO

### Speech Enhancement

We use off-the-shelf speech enhancement tools to enhance the audio files. Specifically, we apply MP-SENet, a speech enhancement model that is trained on the VoiceBank-DEMAND dataset. The model is available on [the official repo](https://github.com/yxlu-0102/MP-SENet).

You should follow the instructions in the repo to install the requirements and download the model. You can use the `apply_mpsenet.py` script to apply the speech enhancement model to the audio files. The script can be used as follows:

```bash
python apply_mpsenet.py --checkpoint_file <path_to_checkpoint> --root_folder <root_folder> --output_folder <output_folder>
```

where `<path_to_checkpoint>` is the path to the MP-SENet checkpoint, `<root_folder>` is the root folder to the extended dataset and `<output_folder>` is the folder where the enhanced audio files will be saved.

The output folder will contain the enhanced audio files following the same structure as the input folder.