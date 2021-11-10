from easydict import EasyDict as edict

opt = edict()

opt.network = edict()
opt.network.emb_size = 512
opt.network.input_shape = (3, 112, 112)

opt.utils = edict()
opt.utils.checkpoint = ""
opt.utils.seed = 1234
opt.utils.num_gpu = 8


opt.dataset = edict()
opt.dataset.num_workers = 2
opt.dataset.trainset = [
        "glint360k",
        "megaface"
        ]


def uniform_dataset(name, rec_path, idx_path, batch_size, **kwargs):
    dataset = edict()
    dataset.rec_path = rec_path
    dataset.idx_path = idx_path
    dataset.batch_size = batch_size
    opt.dataset[name] = dataset

uniform_dataset('glint360k', 
        '/home/users/han.tang/data/public_face_data/glint/glint360k/train.rec', 
        '/home/users/han.tang/data/public_face_data/glint/glint360k/train.idx', 
        batch_size=112)
uniform_dataset('megaface', 
        '/home/users/han.tang/data/public_face_data/faces_megafacetrain_112x112/train.rec', 
        '/home/users/han.tang/data/public_face_data/faces_megafacetrain_112x112/train.idx', 
        batch_size=56)