
"""
Entry point for Semantic Alignment
"""
import sys
import argparse
import logging
from core.trainers.semantic_aligner import SemanticAlignerTrainer

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_path", required=True, help="Path to Student LLM")
    parser.add_argument("--teacher_path", required=True, help="Path to Teacher CLIP")
    parser.add_argument("--data_path", required=True, help="Path to training data text file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()
    
    config = {
        "student_model_path": args.student_path,
        "teacher_model_path": args.teacher_path,
        "train_data_paths": [args.data_path],
        "output_dir": args.output_dir,
        "data_dir": ".", # Dummy
        "max_steps": args.max_steps
    }
    
    trainer = SemanticAlignerTrainer()
    valid, msg = trainer.validate_config(config)
    if not valid:
        logger.error(f"Config Invalid: {msg}")
        sys.exit(1)
        
    logger.info("Starting Semantic Alignment...")
    for progress in trainer.train(config):
        print(f"Step {progress.step}/{progress.total_steps}: Loss={progress.loss:.4f} {progress.message}", flush=True)
        
    logger.info("Done!")

if __name__ == "__main__":
    main()
