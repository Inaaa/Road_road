from model import *
from data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2,'/mrtstorage/users/chli/data_road/training','image_2','gt_image_2',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
testGene = testGenerator("/mrtstorage/users/chli/data_road/testing/image_2")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("/mrtstorage/users/chli/data_road/image",results)

print('!!!!!!!!!!!!!!!!!')