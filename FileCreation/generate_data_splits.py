from glob import glob 
import os 
import pandas as pd
import random
import copy

def get_labels_dataframe(metadata_path):
    # paths 
    f = open(metadata_path, "r")
    original_names = []
    UPDRS = []
    UPDRS_speech = []
    H_Y = []
    SEX = []
    AGE = []
    time_after_diagnosis = []

    for line in f:
        info = line.split(';')
        if "ORIGINAL NAME" in info[0]:
            print("Skipping header")
            continue
        if info[0] == "":
            continue
        original_names.append(info[0])
        UPDRS.append(info[1])
        UPDRS_speech.append(info[2])
        H_Y.append(info[3])
        SEX.append(info[4])
        AGE.append(int(info[5]))
        time_after_diagnosis.append(info[6].strip())
        
    print("Number of original names: ", len(original_names))
    print("Number of UPDRS: ", len(UPDRS))
    print("Number of UPDRS speech: ", len(UPDRS_speech))
    print("Number of H_Y: ", len(H_Y))
    print("Number of SEX: ", len(SEX))
    print("Number of AGE: ", len(AGE))
    print("Number of time after diagnosis: ", len(time_after_diagnosis))

    # print("Original names: ", original_names)
    
    status = []
    # status = PD if the UPDRS is different from empty
    status = ["pd" if u != "" else "hc" for u in UPDRS]
    # print how many PD and HC
    # print("Number of PD: ", status.count("PD"))
    # print("Number of HC: ", status.count("HC"))

    # create a df with those info
    df = pd.DataFrame(columns=["original_name", "UPDRS", "UPDRS_speech", "H_Y", "SEX", "AGE", "time_after_diagnosis", "status"])
    df["original_name"] = original_names
    df["UPDRS"] = UPDRS
    df["UPDRS_speech"] = UPDRS_speech
    df["H_Y"] = H_Y
    df["SEX"] = SEX
    df["AGE"] = AGE
    df["time_after_diagnosis"] = time_after_diagnosis
    df["status"] = status

    print(df.head())
    # print(df.info())
    return df


metadata_path = "/mnt/disk2/mfturco/Data/PC-GITA-CVS.csv"
dataset_folder = "/mnt/disk2/mlaquatra/pc_gita_16khz/"
split_folder = "/mnt/disk2/mlaquatra/pc_gita_splits/"

df_info = get_labels_dataframe(metadata_path)
print(df_info.head())

# in the dataset folder look for all the .wav files even recursively and in all the subfolders
audio_paths = []
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join(root, file)
            audio_paths.append(audio_path)
            
print("Number of audio files: ", len(audio_paths))
print("Number of audio files with words: ", len([a for a in audio_paths if "words" in a or "Words" in a]))
print("Number of audio files with monologue: ", len([a for a in audio_paths if "monologue" in a or "Monologue" in a]))
print("Number of audio files with readtext: ", len([a for a in audio_paths if "readtext" in a or "read_text" in a]))
print("Number of audio files with DDK: ", len([a for a in audio_paths if "DDK" in a]))
print("Number of audio files with sentences: ", len([a for a in audio_paths if "sentences" in a or "Sentences" in a]))


# set random seed
random.seed(42)

def check_overall_overlap(dict_speakers_folders):
    all_speakers = set()
    for v in dict_speakers_folders.values():
        all_speakers.update(v["pd"])
        all_speakers.update(v["hc"])
    
    unique_speakers = set()
    overlap = set()
    for speaker in all_speakers:
        if speaker in unique_speakers:
            overlap.add(speaker)
        else:
            unique_speakers.add(speaker)
    
    if overlap:
        print(f"Overall overlap detected: {overlap}")
    else:
        print("No overall overlap detected")

def get_balanced_patients_groups(df_info, n_groups=10):
    speaker_ids = list(set(df_info["original_name"].tolist()))
    print("Number of unique speaker ids: ", len(speaker_ids))

    pd_speaker_ids = list(set(df_info[df_info["status"] == "pd"]["original_name"].tolist()))
    hc_speaker_ids = list(set(df_info[df_info["status"] == "hc"]["original_name"].tolist()))
    print("Number of unique speaker ids for PD: ", len(pd_speaker_ids))

    male_pd_speaker_ids = list(set(df_info[(df_info["status"] == "pd") & (df_info["SEX"] == "M")]["original_name"].tolist()))
    female_pd_speaker_ids = list(set(df_info[(df_info["status"] == "pd") & (df_info["SEX"] == "F")]["original_name"].tolist()))
    male_hc_speaker_ids = list(set(df_info[(df_info["status"] == "hc") & (df_info["SEX"] == "M")]["original_name"].tolist()))
    female_hc_speaker_ids = list(set(df_info[(df_info["status"] == "hc") & (df_info["SEX"] == "F")]["original_name"].tolist()))
    print("Number of male PD speakers: ", len(male_pd_speaker_ids))
    print("Number of male HC speakers: ", len(male_hc_speaker_ids))
    print("Number of female PD speakers: ", len(female_pd_speaker_ids))
    print("Number of female HC speakers: ", len(female_hc_speaker_ids))

    overall_male_pd_speaker_ids = copy.deepcopy(male_pd_speaker_ids)
    overall_female_pd_speaker_ids = copy.deepcopy(female_pd_speaker_ids)
    overall_male_hc_speaker_ids = copy.deepcopy(male_hc_speaker_ids)
    overall_female_hc_speaker_ids = copy.deepcopy(female_hc_speaker_ids)

    n_folds = 10
    dict_speakers_folders = {}
    for i in range(n_folds):
        dict_speakers_folders[i] = {"pd": [], "hc": []}

    for i in range(n_folds):
        if i % 2 == 0:
            n_males_hc = 3
            n_females_hc = 2
            n_males_pd = 2
            n_females_pd = 3
        else:
            n_males_hc = 2
            n_females_hc = 3
            n_males_pd = 3
            n_females_pd = 2

        male_hc = random.sample(male_hc_speaker_ids, k=n_males_hc)
        male_pd = random.sample(male_pd_speaker_ids, k=n_males_pd)
        female_hc = random.sample(female_hc_speaker_ids, k=n_females_hc)
        female_pd = random.sample(female_pd_speaker_ids, k=n_females_pd)

        male_hc_speaker_ids = [s for s in male_hc_speaker_ids if s not in male_hc]
        male_pd_speaker_ids = [s for s in male_pd_speaker_ids if s not in male_pd]
        female_hc_speaker_ids = [s for s in female_hc_speaker_ids if s not in female_hc]
        female_pd_speaker_ids = [s for s in female_pd_speaker_ids if s not in female_pd]

        dict_speakers_folders[i]["pd"] = male_pd + female_pd
        dict_speakers_folders[i]["hc"] = male_hc + female_hc

    print("Speakers folders: ", dict_speakers_folders)

    for k, v in dict_speakers_folders.items():
        print(f"Group {k}")
        print("PD: ", v["pd"])
        print("HC: ", v["hc"])
        print("Number of PD male: ", len([s for s in v["pd"] if (s in overall_male_pd_speaker_ids)]))
        print("Number of PD female: ", len([s for s in v["pd"] if (s in overall_female_pd_speaker_ids)]))
        print("Number of HC male: ", len([s for s in v["hc"] if (s in overall_male_hc_speaker_ids)]))
        print("Number of HC female: ", len([s for s in v["hc"] if (s in overall_female_hc_speaker_ids)]))
        print("-------------------")
        
    # Final check for overlaps
    check_overall_overlap(dict_speakers_folders)
    return dict_speakers_folders
        
speakers_info = get_balanced_patients_groups(df_info)

for fold in range(10):
    pd_speakers = speakers_info[fold]["pd"]
    hc_speakers = speakers_info[fold]["hc"]
    print(f"Fold {fold+1}")
    
    test_speakers = pd_speakers + hc_speakers
    
    # create folder TRAIN_TEST_{fold}
    os.makedirs(f"{split_folder}/TRAIN_TEST_{fold}", exist_ok=True)
    train_f = f"{split_folder}/TRAIN_TEST_{fold}/train.csv"
    test_f = f"{split_folder}/TRAIN_TEST_{fold}/test.csv"
    
    # test set contains hc_speakers and pd_speakers
    original_names = []
    UPDRS = []
    UPDRS_speech = []
    H_Y = []
    SEX = []
    AGE = []
    time_after_diagnosis = []
    status = []
    filenames = []
    speaker_ids = []
    for speaker in test_speakers:
        speaker_info = df_info[df_info["original_name"] == speaker]
        # get all the files containing the speaker
        speaker_files = [a for a in audio_paths if speaker in a]
        for f in speaker_files:
            original_names.append(speaker_info["original_name"].values[0])
            UPDRS.append(speaker_info["UPDRS"].values[0])
            UPDRS_speech.append(speaker_info["UPDRS_speech"].values[0])
            H_Y.append(speaker_info["H_Y"].values[0])
            SEX.append(speaker_info["SEX"].values[0])
            AGE.append(speaker_info["AGE"].values[0])
            time_after_diagnosis.append(speaker_info["time_after_diagnosis"].values[0])
            status.append(speaker_info["status"].values[0])
            speaker_ids.append(speaker)
            filenames.append(f)

    test_df = pd.DataFrame(columns=["original_name", "UPDRS", "UPDRS-speech", "H_Y", "SEX", "AGE", "time_after_diagnosis", "status", "filename", "speaker_id"])
    test_df["original_name"] = original_names
    test_df["UPDRS"] = UPDRS
    test_df["UPDRS_speech"] = UPDRS_speech
    test_df["H_Y"] = H_Y
    test_df["SEX"] = SEX
    test_df["AGE"] = AGE
    test_df["time_after_diagnosis"] = time_after_diagnosis
    test_df["status"] = status
    test_df["audio_path"] = filenames
    test_df["speaker_id"] = speaker_ids
    
    # train set contains all the other speakers
    train_speakers = [s for s in set(df_info["original_name"].tolist()) if s not in test_speakers]
    original_names = []
    UPDRS = []
    UPDRS_speech = []
    H_Y = []
    SEX = []
    AGE = []
    time_after_diagnosis = []
    status = []
    filenames = []
    speaker_ids = []
    
    for speaker in train_speakers:
        speaker_info = df_info[df_info["original_name"] == speaker]
        # get all the files containing the speaker
        speaker_files = [a for a in audio_paths if speaker in a]
        for f in speaker_files:
            original_names.append(speaker_info["original_name"].values[0])
            UPDRS.append(speaker_info["UPDRS"].values[0])
            UPDRS_speech.append(speaker_info["UPDRS_speech"].values[0])
            H_Y.append(speaker_info["H_Y"].values[0])
            SEX.append(speaker_info["SEX"].values[0])
            AGE.append(speaker_info["AGE"].values[0])
            time_after_diagnosis.append(speaker_info["time_after_diagnosis"].values[0])
            status.append(speaker_info["status"].values[0])
            filenames.append(f)
            speaker_ids.append(speaker)
            
    train_df = pd.DataFrame(columns=["original_name", "UPDRS", "UPDRS-speech", "H_Y", "SEX", "AGE", "time_after_diagnosis", "status", "filename"])
    train_df["original_name"] = original_names
    train_df["UPDRS"] = UPDRS
    train_df["UPDRS_speech"] = UPDRS_speech
    train_df["H_Y"] = H_Y
    train_df["SEX"] = SEX
    train_df["AGE"] = AGE
    train_df["time_after_diagnosis"] = time_after_diagnosis
    train_df["status"] = status
    train_df["audio_path"] = filenames
    train_df["speaker_id"] = speaker_ids
    
    test_df.to_csv(test_f, index=False)
    train_df.to_csv(train_f, index=False)
    print(f"Test file saved at {test_f}")
    print(f"Train file saved at {train_f}")
    
    print("-------------------")
    print(f"For fold {fold+1}")
    print(f"Test set: {len(test_df)}")
    print(f"Train set: {len(train_df)}")
    # print("Number of PD in test set: ", len(test_df[test_df["status"] == "pd"]))
    # print("Number of HC in test set: ", len(test_df[test_df["status"] == "hc"]))
    # print("Number of PD in train set: ", len(train_df[train_df["status"] == "pd"]))
    # print("Number of HC in train set: ", len(train_df[train_df["status"] == "hc"]))
    # print("Number of files containing words in all audio files: ", len([a for a in audio_paths if "words" in a or "Words" in a]))
    # print("Number of files containing words in test set: ", len(test_df[test_df["audio_path"].str.contains("Words")]))
    # # print("files containing words in test set: ", test_df[test_df["audio_path"].str.contains("Words")])
    # print("Number of files containing words in train set: ", len(train_df[train_df["audio_path"].str.contains("Words")]))
    # print("Number of files containing monologue in test set: ", len(test_df[test_df["audio_path"].str.contains("monologue")]))
    # print("Number of files containing readtext in test set: ", len(test_df[test_df["audio_path"].str.contains("readtext")]))
    # print("Number of files containing DDK in test set: ", len(test_df[test_df["audio_path"].str.contains("DDK")]))
    # print("Number of files containing sentences in test set: ", len(test_df[test_df["audio_path"].str.contains("sentences")]))
    