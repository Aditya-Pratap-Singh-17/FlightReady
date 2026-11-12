"""Generate clearly synthetic software fixtures for the connected browser tests.

Run: python -m backend.scripts.connected_research_fixture
No downloads, model execution or empirical research claims.
"""
import argparse
import io
from pathlib import Path
import zipfile
from PIL import Image
from backend.workbench.synthetic import detection_fixture


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('.runtime/flightready-verification'))
    args=parser.parse_args(); directory=args.directory
    directory.mkdir(parents=True,exist_ok=True)
    detection_fixture(directory/'synthetic-drones.zip',seed=731)
    image=io.BytesIO(); Image.new('RGB',(64,64),(40,50,60)).save(image,format='PNG')
    with zipfile.ZipFile(directory/'visdrone-val.zip','w') as archive:
        archive.writestr('VisDrone2019-DET-val/images/software-fixture.png',image.getvalue())
        archive.writestr('VisDrone2019-DET-val/annotations/software-fixture.txt','10,10,20,20,1,4,0,0\n')
    (directory/'FIXTURES.txt').write_text('Generated synthetic software fixtures. Not real UAV imagery or empirical research evidence.\n',encoding='utf-8')
    print('Synthetic fixtures created at '+str(directory.resolve()))


if __name__=='__main__': main()
