"""Regenerate small SYNTHETIC datasets; these do not validate real UAV research."""
from pathlib import Path
import io
import zipfile
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]/'.runtime'/'demo_data'

def generate(output=None):
    root = Path(output) if output is not None else ROOT
    root.mkdir(parents=True, exist_ok=True)
    samples = {
        'clean.csv':'x,label,altitude_meters,lighting,geography_type,split,flight_id\n1,car,30,daytime,urban,train,A\n2,pedestrian,40,daytime,urban,train,A\n3,car,50,daytime,urban,test,B\n4,pedestrian,60,daytime,urban,test,B\n',
        'missing_required.csv':'x,lighting\n1,daytime\n2,daytime\n',
        'invalid_values.csv':'x,label,altitude_meters\n1,car,-50\n2,pedestrian,infinity\n',
        'duplicates.csv':'x,label\n1,car\n1,car\n2,pedestrian\n',
        'imbalanced.csv':'x,label\n'+''.join(f'{i},car\n' for i in range(20))+'20,pedestrian\n',
        'unknown_scenarios.csv':'x,label\n1,car\n2,pedestrian\n',
    }
    for name,content in samples.items():
        (root/name).write_text(content,encoding='utf-8')
    for invalid in (False,True):
        name = 'detection_invalid.zip' if invalid else 'detection_valid.zip'
        with zipfile.ZipFile(root/name,'w') as z:
            z.writestr('classes.txt','car\npedestrian\n')
            metadata = 'image,lighting,altitude_meters,geography_type,split,flight_id\n'
            for i in range(4):
                # Deliberately simple generated drawings, not UAV photographs.
                image = Image.new('RGB',(128,96),(60+i*20,90,65))
                draw = ImageDraw.Draw(image)
                draw.rectangle((48,36,80,60),fill=(190,175-i*20,60))
                draw.text((4,4),'SYNTHETIC',fill='white')
                output=io.BytesIO();image.save(output,format='PNG')
                image_name=f'images/example_{i}.png'
                z.writestr(image_name,output.getvalue())
                box='0.1 0.5 0.8 0.4' if invalid and i==0 else '0.5 0.5 0.25 0.25'
                z.writestr(f'labels/example_{i}.txt',f'{i%2} {box}\n')
                metadata+=f'{image_name},daytime,{30+i*10},urban,{"train" if i<2 else "test"},{"A" if i<2 else "B"}\n'
            z.writestr('metadata.csv',metadata)
    print(f'Created {len(samples)+2} synthetic demonstration datasets in {root}')

if __name__=='__main__':generate()
