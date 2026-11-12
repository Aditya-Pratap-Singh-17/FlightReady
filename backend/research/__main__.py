"""Run `python -m backend.research --help`. No command launches mass training."""
import argparse
import json
from pathlib import Path
from backend.context import DatasetContext
from backend.research.variants import VariantConfig,create_variant
from backend.research.experiments import TrainingConfig,assess_variant,train_detector,export_experiments

def main():
    parser=argparse.ArgumentParser(description='Optional controlled UAV experiments')
    sub=parser.add_subparsers(dest='command',required=True)
    variant=sub.add_parser('variant');variant.add_argument('parent');variant.add_argument('output');variant.add_argument('config')
    assess=sub.add_parser('assess');assess.add_argument('directory');assess.add_argument('context')
    train=sub.add_parser('train');train.add_argument('directory');train.add_argument('config');train.add_argument('--execute',action='store_true')
    export=sub.add_parser('export');export.add_argument('destination');export.add_argument('directories',nargs='+')
    args=parser.parse_args()
    if args.command=='variant':
        result=create_variant(args.parent,args.output,VariantConfig.model_validate_json(Path(args.config).read_text()))
        print(result['variant_id'])
    elif args.command=='assess':
        result=assess_variant(args.directory,DatasetContext.model_validate_json(Path(args.context).read_text()))
        print(result['variant_id'])
    elif args.command=='train':
        result=train_detector(args.directory,TrainingConfig.model_validate_json(Path(args.config).read_text()),execute=args.execute)
        print(result['variant_id'])
    else:
        rows=export_experiments(args.directories,args.destination);print(f'Exported {len(rows)} rows')

if __name__=='__main__':main()
