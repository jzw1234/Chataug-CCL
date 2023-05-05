import json
import os
from transformers import AutoTokenizer
import argparse
from PIL import Image
import torch
from CLIP import clip
import utils
JSONL_FILE_DIR = "../jsonl_files"
TOKENIED_FILE_DIR = "../tokenized_files"
annotation_dir = "new_annotation.json"
from torchvision import transforms
import matplotlib.pyplot as plt
from dall_e import map_pixels, unmap_pixels, load_model
import torchvision.transforms as T


T5_PROMPT = {"wiki_bio": "convert the table to text: ",
             "totto_meta": "",
             "common_gen": "generate a sentence with: ",
             "multi_news": "summarize: ",
             "xsum": "summarize: ",
             "wmt16_ro-en": "translate Romanian to English: ",
             "java": "<java> ",
             "python": "<python> "
             }


def tokenize_raw(ds_name,  ptm_alias="t5", prompt=""):
    """
    ds_name: file name of raw data
    model: the pretrained model used
    ptm_alias: alias for the model
    prompt: optional, prompt for t5-based model
    """
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_vae = utils.create_d_vae(
        weight_path="DVAE_weight", d_vae_type="dall-e",
        device=device, image_size=224)
    # base_dir = f"{JSONL_FILE_DIR}/{ds_name}"
    tokenized_dir = f"{TOKENIED_FILE_DIR}/{ds_name}"

    # model, preprocess = clip.load("ViT-B/16", device=device)
    # if not os.path.exists(tokenized_dir):
    #     os.makedirs(tokenized_dir)


    files = ["val", "train"]
    files_tokenized = [f"val.{ptm_alias}.jsonl", f"train.{ptm_alias}.jsonl"]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    insts_list = []
    ann=json.loads(open(annotation_dir,encoding='utf-8').read())
    flag_file=json.loads(open('front_slide.json',encoding='utf-8').read())
    for split in files:
        insts = []
        for line in ann[split]:
            if len(insts)>50:
                break
            insts.append(
                        {
                            "image_path":line["image_path"],
                            "report":line["report"],
                            "flag":flag_file[line["image_path"][0]]
                        }
                        )
        insts_list.append(insts)
    for i, insts in enumerate(insts_list):
        for inst in insts:
            # if "t5" in model:
            #     source = prompt + inst["image_path"]
            # else:
            if inst["flag"]==0:
                source = inst["image_path"][0]
            else:source = inst["image_path"][1]
            target = inst["report"]
            img_path1=os.path.join("I:/project/pythonProject2/data/iu_xray/images",source)
            image_path1=transform(Image.open(img_path1).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                input_ids = d_vae.get_codebook_indices(image_path1).flatten(1).squeeze(0).cpu()
                recover=d_vae.decode(d_vae.get_codebook_indices(image_path1))
                # plt.imshow(recover.cpu().numpy())
                # x_rec = unmap_pixels(torch.sigmoid(recover[:, :3]))
                # x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
                # plt.imshow(x_rec)
                # plt.show()

            # image1 = preprocess(image_path1).unsqueeze(0).to(device)
            # with torch.no_grad():
            #     image_features1 = model.encode_image(image1)
            # src_id = image_features1.squeeze().cpu().numpy()#tokenizer.encode(source)
            tgt_id = tokenizer.encode(target)
            # print(len(input_ids))
            inst["src_id"] = input_ids.numpy().tolist()
            inst["tgt_id"] = tgt_id
        print("write into ... ", os.path.join(tokenized_dir, files_tokenized[i]))
        with open(os.path.join(tokenized_dir, files_tokenized[i]), "w",encoding='utf-8') as f:
            for inst in insts:
                print(json.dumps(inst, ensure_ascii=False), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True,
                        help=" the pretrained model used, tokenizer.from_pretrained(model_name)")
    parser.add_argument('--dataset', required=True, help="selected dataset")
    parser.add_argument('--ptm', default=None, help=" mark the tokenized file")

    args = parser.parse_args()
    if not args.ptm:
        args.ptm = args.model_name.split("/")[-1].split("-")[0]
        print("You are using the pretrain model: ", args.ptm)
    # if you need a prompt
    prompt = ""
    if "t5" in args.model_name:
        prompt = T5_PROMPT[args.dataset]

    tokenize_raw(args.dataset,  args.ptm, prompt)
