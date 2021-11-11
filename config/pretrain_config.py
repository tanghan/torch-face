from easydict import EasyDict as edict

opt = edict()

opt.network = edict()
opt.network.emb_size = 512
opt.network.input_shape = (3, 112, 112)

opt.utils = edict()
opt.utils.checkpoint = "/job_data/"
opt.utils.seed = 1234
opt.utils.num_gpu = 8
opt.utils.num_epoch = 20
opt.utils.fp16 = True


opt.dataset = edict()
opt.dataset.num_workers = 2
opt.dataset.trainset = [
        "glint360k",
        "megaface"
        ]


def uniform_dataset(name, rec_path, idx_path, batch_size, num_samples, num_classes, **kwargs):
    dataset = edict()
    dataset.rec_path = rec_path
    dataset.idx_path = idx_path
    dataset.batch_size = batch_size
    dataset.num_samples = num_samples
    dataset.num_classes = num_classes
    opt.dataset[name] = dataset

uniform_dataset('glint360k', 
        '/cluster_home/data/public_face_data/glint360k/train.rec',
        '/cluster_home/data/public_face_data/glint360k/train.idx', 
        batch_size=120,
        num_samples=17091657,
        num_classes=360232)
uniform_dataset('megaface', 
        '/cluster_home/data/public_face_data/megaface/train.rec', 
        '/cluster_home/data/public_face_data/megaface/train.idx', 
        batch_size=40,
        num_samples=4574213,
        num_classes=657078)
