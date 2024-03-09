from glob import glob 
import os 
import pandas as pd
import sys
import random

# PATHS 
GENERAL_FOLDER = "/mnt/disk2/mfturco/Data/"
PC_GITA_DOWNSAMPLED = GENERAL_FOLDER+"PC-GITA_downsampled_16000Hz/"
INPUT_DIR = PC_GITA_DOWNSAMPLED+"DDK_analysis/"
OUTPUT_DIR_1 = PC_GITA_DOWNSAMPLED+"DDK_analysis_audio_files_path_16kHz.csv"
OUTPUT_DIR_2 = PC_GITA_DOWNSAMPLED+"OVERALL_audio_files_path_16kHz.csv"
OUTPUT_DIR_3 = PC_GITA_DOWNSAMPLED+"GROUPED_CONTROLS+PATIENTS.csv"


# Creation of a csv file with all paths of the audio files (.wav) 
list = []
for root,dirs,files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith("wav"):
            audio_path = os.path.join(root,file)
            person_status = audio_path.split("/")[-2] #-3 for Words while the other tasks -2
            list.append([audio_path,person_status])
            dataset = pd.DataFrame(list)
            dataset.to_csv(OUTPUT_DIR_1,header=['audio_path','status'],index=False)

#Creation of a cumulative csv file (with inside each task)
files = glob(PC_GITA_DOWNSAMPLED+'*.csv')
df = pd.DataFrame()
for file in files: 
    data = pd.read_csv(file,delimiter=',')
    df = pd.concat([df,data],axis=0)
df.to_csv(OUTPUT_DIR_2,index=False)

#Group allocation: the 100 speakers will be split in 10 groups with 5 PD and 5 HC per group
#with the idea of gender balance in each groups more specifically:
# if group is even : for HC (3 male , 2 female) for PD (2 male, 3 female) = 5 male , 5 female
# if group is odd:   for HC (2 male, 3 female) for PD ( 3 male, 2 female) = 5 male, 5 female
random.seed(10)
df = pd.read_csv(OUTPUT_DIR_2)
original_df = pd.read_csv(GENERAL_FOLDER+'PC-GITA-CVS.csv',delimiter=";")
df.loc[df['status']== 'Control','status'] = 'hc'
df.loc[df['status']== 'HC','status'] = 'hc'
df.loc[df['status']== 'Patologica','status'] = 'pd'
df.loc[df['status']== 'PD','status'] = 'pd'
general_ids = []
for ind in df.index:
    file_name = os.path.basename(os.path.normpath(df['audio_path'][ind]))[:-4]#for removing .npy extension
    general_ids.append(file_name) #id audio recordings
df.insert(0,'original_id',general_ids)
speaker_ids = []
for ind in df.index:
    if df['status'][ind] == 'pd':
        speaker_ids.append(df['original_id'][ind][0:13])
    else:
        speaker_ids.append(df['original_id'][ind][0:14])
df['speaker_id'] = speaker_ids
df['group'] = 0
df2 = pd.merge(df,original_df,left_on='speaker_id',right_on='RECODING ORIGINAL NAME')
df2.drop(columns=['RECODING ORIGINAL NAME'],inplace=True)
df = df2
patient_df = df[df['status']=='pd']
control_df = df[df['status']=='hc']
# PD (Parkinsonian People)
# Building 2 dataset: one with males only while the other females only
males_pd = patient_df.loc[patient_df['SEX'] == 'M']
females_pd = patient_df.loc[patient_df['SEX'] == 'F']
#Number of groups for which i have to split people
GROUPS_NUMBER = 10
# Creation of 2 lists in which i put the speaker id one for male and one for female
list_speaker_id_m = males_pd['speaker_id'].unique()
list_speaker_id_m= list_speaker_id_m.tolist()
list_speaker_id_f = females_pd['speaker_id'].unique()
list_speaker_id_f= list_speaker_id_f.tolist()
for group in range(1,GROUPS_NUMBER+1):
    #EVEN
    if group %2 == 0:
        print(f'Group',group)
        #randomly sample 2 males from the speaker id list
        y = random.sample(list_speaker_id_m,k=2)
        #find them inside the dataset
        patient_df.loc[patient_df['speaker_id'].isin(y),'group'] = group
        #drop them from the list from which i will sample again during the next cicle
        y, list_speaker_id_m = [i for i in y if i not in list_speaker_id_m], [j for j in list_speaker_id_m if j not in y]
        #randomly sample 3 females from the speaker id list
        x = random.sample(list_speaker_id_f,k=3)
        #find them inside the dataset
        patient_df.loc[patient_df['speaker_id'].isin(x),'group'] = group
        #drop them from the list from which i will sample again during the next cicle
        x, list_speaker_id_f = [i for i in x if i not in list_speaker_id_f], [j for j in list_speaker_id_f if j not in x]
    #ODD
    else:
        print(f'Group',group)
        #randomly sample 3 males from the speaker id list
        y = random.sample(list_speaker_id_m,k=3)
        patient_df.loc[patient_df['speaker_id'].isin(y),'group'] = group
        y, list_speaker_id_m = [i for i in y if i not in list_speaker_id_m], [j for j in list_speaker_id_m if j not in y]
        #randomly sample 2 females from the speaker id list
        x = random.sample(list_speaker_id_f,k=2) 
        patient_df.loc[patient_df['speaker_id'].isin(x),'group'] = group
        x, list_speaker_id_f = [i for i in x if i not in list_speaker_id_f], [j for j in list_speaker_id_f if j not in x]
#HC
# Building 2 dataset: one with males only while the other females only
males_hc = control_df.loc[control_df['SEX'] == 'M']
females_hc = control_df.loc[control_df['SEX'] == 'F']
# Creation of 2 lists in which i put the speaker id one for male and one for female
list_speaker_id_m = males_hc['speaker_id'].unique()
list_speaker_id_m= list_speaker_id_m.tolist()
list_speaker_id_f = females_hc['speaker_id'].unique()
list_speaker_id_f= list_speaker_id_f.tolist()
for group in range(1,GROUPS_NUMBER+1):
    if group %2 == 0:
        y = random.sample(list_speaker_id_m,k=3)
        control_df.loc[control_df['speaker_id'].isin(y),'group'] = group
        y, list_speaker_id_m = [i for i in y if i not in list_speaker_id_m], [j for j in list_speaker_id_m if j not in y]
        x = random.sample(list_speaker_id_f,k=2)
        control_df.loc[control_df['speaker_id'].isin(x),'group'] = group
        x, list_speaker_id_f = [i for i in x if i not in list_speaker_id_f], [j for j in list_speaker_id_f if j not in x]
    else:
        y = random.sample(list_speaker_id_m,k=2)
        control_df.loc[control_df['speaker_id'].isin(y),'group'] = group
        y, list_speaker_id_m = [i for i in y if i not in list_speaker_id_m], [j for j in list_speaker_id_m if j not in y]
        x = random.sample(list_speaker_id_f,k=3)
        control_df.loc[control_df['speaker_id'].isin(x),'group'] = group
        x, list_speaker_id_f = [i for i in x if i not in list_speaker_id_f], [j for j in list_speaker_id_f if j not in x]
speaker_grouped = pd.concat([patient_df,control_df])
speaker_grouped.to_csv(OUTPUT_DIR_3,index=False)



#Lastly, creation of 10 folders for train and test the NN models 
OUTPUT_DIR = '/mnt/disk2/mfturco/Data/NN_TRAIN_TESTS_16KHz'
df = pd.read_csv('/mnt/disk2/mfturco/Data/PC-GITA_downsampled_16000Hz/GROUPED_CONTROLS+PATIENTS.csv')
for group in range(1,10+1):
    # Directory  
    directory = "TRAIN_TEST_"+str(group)
    # Parent Directory path  
    parent_dir = OUTPUT_DIR
    path = os.path.join(parent_dir, directory)   
    os.mkdir(path)  
    print("Directory '% s' created" % directory) 

for group in range(1,10+1):
    test = df.loc[df['group']== group]
    train = pd.concat([df,test]).drop_duplicates(keep=False)
    output_dir_grouped = OUTPUT_DIR+"/TRAIN_TEST_"+str(group)
    test.to_csv(output_dir_grouped+'/test.csv',index = False)
    train.to_csv(output_dir_grouped+'/train.csv',index = False)