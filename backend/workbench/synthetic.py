"""Explicitly synthetic detector fixture; never evidence of real UAV performance."""
from pathlib import Path
import io
import zipfile
import numpy as np
from PIL import Image, ImageDraw


def detection_fixture(path, seed=1, external=False):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata=['image,split,flight_id,source']
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as output:
        output.writestr('classes.txt','drone\n')
        for index in range(4 if external else 10):
            split='test' if external else 'train' if index<6 else 'val' if index<8 else 'test'
            rng=np.random.default_rng(seed*100+index)
            image=Image.fromarray(rng.integers(0,180,(64,64,3),dtype=np.uint8))
            draw=ImageDraw.Draw(image)
            draw.rectangle((20,24,44,40),fill=(225,225,225))
            draw.line((16,20,48,44),fill=(240,240,240),width=3)
            raw=io.BytesIO();image.save(raw,format='PNG')
            name=f'images/{split}/{index}.png'
            output.writestr(name,raw.getvalue())
            output.writestr(f'labels/{split}/{index}.txt','0 .5 .5 .5 .375\n')
            metadata.append(f'{name},{split},synthetic-{seed}-{index},explicitly synthetic software test')
        output.writestr('metadata.csv','\n'.join(metadata)+'\n')
    return path
