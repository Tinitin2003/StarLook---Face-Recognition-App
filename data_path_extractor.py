import os
import pickle
actors=os.listdir('../face_dataset')
filenames = []
for actor in actors:
    for file in os.listdir(os.path.join('../face_dataset',actor)):
        if os.path.isfile(os.path.join('../face_dataset',actor,file)):
            filenames.append(os.path.join('../face_dataset',actor,file))
# actors=os.listdir('../face_dataset/dataset')
# for actor in actors:
#     for file in os.listdir(os.path.join('../face_dataset/dataset',actor)):
#         filenames.append(os.path.join('../face_dataset/dataset',actor,file))
pickle.dump(filenames,open('filenames.pkl','wb'))
# filenames = []
# for actor in actors:
#     for file in os.listdir(os.path.join('../face_dataset/dataset',actor)):
#         filenames.append(os.path.join('../face_dataset/dataset',actor,file))
# pickle.dump(filenames,open('unclear_face.pkl','wb'))