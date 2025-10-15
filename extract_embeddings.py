import os, json, argparse, numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm

def get_transform():
    return T.Compose([
        T.Resize((224,224)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def load_image(path):
    return Image.open(path).convert('RGB')

def main(data_dir='data', out_dir='.', batch_size=16, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # remove classifier
    model.eval().to(device)
    transform = get_transform()

    # gather image files
    exts = ('.jpg','.jpeg','.png','.webp')
    files = []
    for root, dirs, fs in os.walk(data_dir):
        for f in fs:
            if f.lower().endswith(exts):
                files.append(os.path.join(root,f))
    files = sorted(files)
    if len(files)==0:
        print("No images found in", data_dir)
        return

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(files), batch_size), desc="Extracting embeddings"):
            batch_files = files[i:i+batch_size]
            imgs = [transform(load_image(p)) for p in batch_files]
            x = torch.stack(imgs).to(device)
            feats = model(x).cpu().numpy()
            embeddings.append(feats)

    embeddings = np.vstack(embeddings)
    np.save(os.path.join(out_dir,'embeddings.npy'), embeddings)

    # save relative paths to metadata.json
    # save relative paths
    relative_files = [os.path.relpath(f, data_dir) for f in files]
    with open(os.path.join(out_dir,'metadata.json'),'w') as f:
        json.dump(relative_files, f, indent=2)


    print(f"Saved embeddings.npy and metadata.json for {len(relative_files)} images to {out_dir}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--out_dir', default='.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir, batch_size=args.batch_size, device=args.device)
