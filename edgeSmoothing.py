import glob
from dealMask import dealMask
import cv2
from mixup import mixup

def openfiles(mask_path, output_path):
    masks = glob.glob(mask_path + "*")
    gens = glob.glob(output_path + "gen/*")
    oris = glob.glob(output_path + "real/*")
    return masks, gens, oris

def edgeSmoothing(merge_path, masks, gens, oris):
    for i in range(len(masks)):
        name = masks[i].split("\\")[-1]

        dealed_mask = dealMask(masks[i])
        gen = cv2.imread(gens[i])
        ori = cv2.imread(oris[i])
        print(name)
        print(dealed_mask.shape)
        print(gen.shape)
        print(ori.shape)
        print("---------------")

        merge = mixup(dealed_mask, gen, ori)
        cv2.imwrite(merge_path + name, merge)



if __name__ == '__main__':
    mask_path = 'mask_png/'
    output_path = "201904011442_201903162133_full_model_icme_sa_bn_512_rect_vip_pretrain/val_99_whole/"
    merge_path = 'merge/'
    masks, gens, oris = openfiles(mask_path, output_path)
    masks.sort()
    gens.sort()
    oris.sort()

    assert len(masks) == len(gens) == len(oris)

    edgeSmoothing(merge_path, masks, gens, oris)





